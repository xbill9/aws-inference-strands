#!/usr/bin/env bash
# Launch the demo backends. Every one starts billing; stage/down.sh stops all of them.
#
#   stage/up.sh                 # all five: g6 inf2 trn1 g5g sm
#   stage/up.sh g6 sm           # only some
#
# Launch to healthy, measured 2026-10-04/05: g6 14 min, inf2 14 min, trn1 ~10 min
# (2 min after boot), SageMaker 8 min, g5g 22 min (first weight load from a fresh
# EBS volume). Start at least 30 minutes before you need them.
#
# Instances are tagged Name=strands-demo-<backend>, ManagedBy=strands-demo-run.
# A backend that is already pending or running (in any US region) is left alone.
# When a zone has no capacity the launch tries the region's next default subnet,
# then the next region in that backend's list.
set -uo pipefail
export AWS_PAGER=

PROFILE=${PROFILE:-g6-vllm-instance-profile}
# AMIs are looked up by name in each region, newest first; the g5g image is our own, in us-east-1 only.
G6_AMI_NAME=${G6_AMI_NAME:-Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)*}
NEURON_AMI_NAME=${NEURON_AMI_NAME:-Deep Learning AMI Neuron (Ubuntu 24.04)*}
G5G_AMI=${G5G_AMI:-ami-0b44b90b3d02430ee}    # gpu-vllm-g5g-2b-sm75-vllm0272rc0-80g-v2 (patched vLLM, weights baked in)
SG_NAME=${SG_NAME:-strands-demo-ssm-only}    # no inbound rules; created on first use and kept
# Regions to try in order when a zone has no capacity. Quota is not capacity.
G6_REGIONS=${G6_REGIONS:-us-east-2 us-east-1 us-west-2}
INF2_REGIONS=${INF2_REGIONS:-us-east-2 us-east-1 us-west-2}
TRN1_REGIONS=${TRN1_REGIONS:-us-east-2 us-east-1 us-west-2}
REPACK=${REPACK:-xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4}
NEURON_IMAGE=${NEURON_IMAGE:-docker.io/xbill9/gemma4-optb:slim}
SM_DIR=${SM_DIR:-$HOME/sagemaker-gemma}
SM_ENDPOINT=${SM_ENDPOINT:-gemma-4-e2b-strands}
SM_REGION=${SM_REGION:-us-east-2}

userdata_g6() { cat <<UD
#!/usr/bin/env bash
set -euxo pipefail
systemctl enable --now docker || (apt-get update && apt-get install -y docker.io && systemctl enable --now docker)
docker run -d --name vllm --restart unless-stopped --gpus all --ipc=host -p 8000:8000 \\
  vllm/vllm-openai:v0.30.0 \\
  --model $REPACK \\
  --max-model-len 8192 --gpu-memory-utilization 0.90 \\
  --enable-auto-tool-choice --tool-call-parser gemma4
UD
}

userdata_neuron() { cat <<UD
#!/usr/bin/env bash
set -euxo pipefail
if ! swapon --show --noheadings | grep -q /swapfile; then
  fallocate -l 16G /swapfile; chmod 600 /swapfile; mkswap /swapfile; swapon /swapfile
fi
command -v docker >/dev/null || (apt-get update && apt-get install -y docker.io)
systemctl enable --now docker
docker run -d --name gemma4-neuron --restart unless-stopped --ipc=host --device=/dev/neuron0 \\
  -v gemma4-data:/data -p 8080:8080 $NEURON_IMAGE
UD
}

# The baked g5g image starts vLLM from /opt/serve.sh at boot; add the tool-call
# flags before the weights have loaded, then restart it once.
userdata_g5g() { cat <<'UD'
#!/usr/bin/env bash
set -euxo pipefail
if ! grep -q enable-auto-tool-choice /opt/serve.sh; then
  sed -i 's|  --host 0.0.0.0 --port 8000|  --enable-auto-tool-choice --tool-call-parser gemma4 \\\n  --host 0.0.0.0 --port 8000|' /opt/serve.sh
  bash -n /opt/serve.sh
  systemctl restart vllm.service
fi
UD
}

running() {  # name -> "region instance-id" for one pending or running, in any US region
  local r id
  for r in us-east-1 us-east-2 us-west-1 us-west-2; do
    id=$(aws ec2 describe-instances --region "$r" \
      --filters "Name=tag:Name,Values=$1" Name=instance-state-name,Values=pending,running \
      --query 'Reservations[].Instances[0].InstanceId' --output text)
    [ -n "$id" ] && { echo "$r $id"; return 0; }
  done
}

ami_in() {  # region name-pattern|ami-id -> newest Amazon AMI with that name
  case $2 in ami-*) echo "$2"; return ;; esac
  aws ec2 describe-images --region "$1" --owners amazon --filters "Name=name,Values=$2" Name=state,Values=available \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text
}

sg_in() {  # region -> id of the SSM-only group, created with no inbound rules if missing
  local sg vpc
  # Reuse any SSM-only group already there, including the ones earlier runs made.
  sg=$(aws ec2 describe-security-groups --region "$1" \
       --filters "Name=group-name,Values=$SG_NAME,gpu-vllm-g6-ssm-only,strands-demo-g5g-ssm-only" \
       --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$sg" = None ]; then
    vpc=$(aws ec2 describe-vpcs --region "$1" --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
    sg=$(aws ec2 create-security-group --region "$1" --vpc-id "$vpc" --group-name "$SG_NAME" \
         --description "Strands demo: no inbound, reached through SSM" --query GroupId --output text)
  fi
  echo "$sg"
}

launch() {  # name type ami-name-or-id userdata-fn disk-gb regions...
  local name=$1 type=$2 ami=$3 ud=$4 disk=$5; shift 5
  local up region ami_id sg subnet out udfile
  up=$(running "$name")
  if [ -n "$up" ]; then echo "$name: already up ($up)"; return 0; fi
  udfile=$(mktemp); "$ud" > "$udfile"
  for region in "$@"; do
    ami_id=$(ami_in "$region" "$ami"); sg=$(sg_in "$region")
    for subnet in $(aws ec2 describe-subnets --region "$region" --filters Name=default-for-az,Values=true \
                    --query 'sort_by(Subnets,&AvailabilityZone)[].SubnetId' --output text); do
      out=$(aws ec2 run-instances --region "$region" --image-id "$ami_id" --instance-type "$type" \
        --subnet-id "$subnet" --security-group-ids "$sg" --iam-instance-profile "Name=$PROFILE" \
        --metadata-options HttpTokens=required \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$disk,VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data "file://$udfile" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name},{Key=ManagedBy,Value=strands-demo-run}]" \
        --query 'Instances[0].[InstanceId,Placement.AvailabilityZone]' --output text 2>&1)
      case "$out" in
        i-*) echo "$name: launched $out"; rm -f "$udfile"; return 0 ;;
        *InsufficientInstanceCapacity*|*Unsupported*) echo "$name: no capacity in $region ($subnet), trying the next zone" ;;
        *) echo "$name: $region: $out" | head -2 ;;
      esac
    done
  done
  rm -f "$udfile"; echo "$name: no capacity in any of: $*"; return 1
}

sagemaker() {
  local st
  st=$(aws sagemaker describe-endpoint --region "$SM_REGION" --endpoint-name "$SM_ENDPOINT" \
       --query EndpointStatus --output text 2>/dev/null)
  if [ -n "$st" ]; then echo "sm: endpoint $SM_ENDPOINT already exists ($st)"; return 0; fi
  # Call sm.py directly: the Makefile exports its .env over these values.
  echo "sm: deploying $SM_ENDPOINT in $SM_REGION (InService in ~8 min)"
  ( cd "$SM_DIR" && AWS_REGION=$SM_REGION MODEL_ID=$REPACK ENDPOINT_NAME=$SM_ENDPOINT \
      INSTANCE_TYPE=ml.g6.xlarge SM_VLLM_ENABLE_AUTO_TOOL_CHOICE=true SM_VLLM_TOOL_CALL_PARSER=gemma4 \
      python3 sm.py deploy )
}

want=${*:-g6 inf2 trn1 g5g sm}
pids=()
for b in $want; do
  case $b in
    g6)   launch strands-demo-g6   g6.xlarge    "$G6_AMI_NAME"     userdata_g6     100 $G6_REGIONS & ;;
    inf2) launch strands-demo-inf2 inf2.xlarge  "$NEURON_AMI_NAME" userdata_neuron 100 $INF2_REGIONS & ;;
    trn1) launch strands-demo-trn1 trn1.2xlarge "$NEURON_AMI_NAME" userdata_neuron 200 $TRN1_REGIONS & ;;
    g5g)  launch strands-demo-g5g  g5g.2xlarge  "$G5G_AMI"         userdata_g5g    80  us-east-1 & ;;
    sm)   sagemaker & ;;
    *) echo "unknown backend: $b (use g6 inf2 trn1 g5g sm)"; continue ;;
  esac
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo; echo "Next: stage/forward.sh (once the instances are up), then: source stage/demo.env && python3 stage/check.py"
exit $rc
