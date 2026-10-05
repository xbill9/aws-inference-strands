#!/usr/bin/env bash
# Open an SSM port forward to every running demo instance and write stage/demo.env.
#
#   stage/forward.sh && source stage/demo.env
#
# localhost:8001 -> g6:8000   localhost:8002 -> inf2:8080
# localhost:8003 -> trn1:8080 localhost:8004 -> g5g:8000
# Re-running it closes the old forwards first. Needs the session-manager-plugin.
set -uo pipefail
export AWS_PAGER=
here=$(cd "$(dirname "$0")" && pwd)
pidfile=$here/.forwards.pid
REPACK=${REPACK:-xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4}
SM_ENDPOINT=${SM_ENDPOINT:-gemma-4-e2b-strands}
SM_REGION=${SM_REGION:-us-east-2}

if [ -f "$pidfile" ]; then
  while read -r p; do kill "$p" 2>/dev/null; done < "$pidfile"
  rm -f "$pidfile"
fi

env_lines=("export BEDROCK_MODEL=\${BEDROCK_MODEL:-us.amazon.nova-micro-v1:0}"
           "export BEDROCK_REGION=\${BEDROCK_REGION:-us-east-1}")

forward() {  # backend remote-port local-port; finds the instance in any US region
  local r id="" ping
  for r in us-east-2 us-east-1 us-west-2 us-west-1; do
    id=$(aws ec2 describe-instances --region "$r" \
         --filters "Name=tag:Name,Values=strands-demo-$1" Name=instance-state-name,Values=running \
         --query 'Reservations[0].Instances[0].InstanceId' --output text)
    [ -n "$id" ] && [ "$id" != None ] && break
    id=""
  done
  if [ -z "$id" ]; then echo "$1: not running, skipped"; return 1; fi
  ping=$(aws ssm describe-instance-information --region "$r" --filters "Key=InstanceIds,Values=$id" \
         --query 'InstanceInformationList[0].PingStatus' --output text)
  if [ "$ping" != Online ]; then echo "$1: $id is not on SSM yet ($ping), skipped"; return 1; fi
  aws ssm start-session --region "$r" --target "$id" --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"$2\"],\"localPortNumber\":[\"$3\"]}" > "$here/.fwd-$1.log" 2>&1 &
  echo $! >> "$pidfile"
  echo "$1: localhost:$3 -> $id:$2 ($r)"
}

forward g6   8000 8001 && env_lines+=("export G6_URL=http://localhost:8001/v1 G6_MODEL=$REPACK")
forward inf2 8080 8002 && env_lines+=("export INF2_URL=http://localhost:8002/v1")
forward trn1 8080 8003 && env_lines+=("export TRN1_URL=http://localhost:8003/v1")
forward g5g  8000 8004 && env_lines+=("export G5G_URL=http://localhost:8004/v1 G5G_MODEL=google/gemma-4-E2B-it")
st=$(aws sagemaker describe-endpoint --region "$SM_REGION" --endpoint-name "$SM_ENDPOINT" \
     --query EndpointStatus --output text 2>/dev/null)
if [ "$st" = InService ]; then
  echo "sm: $SM_ENDPOINT InService"
  env_lines+=("export SM_ENDPOINT=$SM_ENDPOINT SM_REGION=$SM_REGION SM_MODEL=$REPACK")
else
  echo "sm: $SM_ENDPOINT ${st:-not found}, skipped"
fi

printf '%s\n' "${env_lines[@]}" > "$here/demo.env"
sleep 5
echo; echo "Wrote stage/demo.env. Next: source stage/demo.env && python3 stage/check.py"
