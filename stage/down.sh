#!/usr/bin/env bash
# Stop everything the demo bills for, then sweep all four US regions.
# Keeps the security groups and the instance profile for the next run.
set -uo pipefail
export AWS_PAGER=
here=$(cd "$(dirname "$0")" && pwd)
SM_DIR=${SM_DIR:-$HOME/sagemaker-gemma}
SM_ENDPOINT=${SM_ENDPOINT:-gemma-4-e2b-strands}
SM_REGION=${SM_REGION:-us-east-2}
REGIONS="us-east-1 us-east-2 us-west-1 us-west-2"

if [ -f "$here/.forwards.pid" ]; then
  while read -r p; do kill "$p" 2>/dev/null; done < "$here/.forwards.pid"; rm -f "$here/.forwards.pid"
fi

for r in $REGIONS; do
  ids=$(aws ec2 describe-instances --region "$r" \
        --filters Name=tag:ManagedBy,Values=strands-demo-run Name=instance-state-name,Values=pending,running,stopping,stopped \
        --query 'Reservations[].Instances[].InstanceId' --output text)
  if [ -n "$ids" ]; then
    echo "$r: terminating $ids"
    aws ec2 terminate-instances --region "$r" --instance-ids $ids --query 'TerminatingInstances[].InstanceId' --output text >/dev/null
  fi
done

if aws sagemaker describe-endpoint --region "$SM_REGION" --endpoint-name "$SM_ENDPOINT" >/dev/null 2>&1; then
  echo "sm: destroying $SM_ENDPOINT"
  ( cd "$SM_DIR" && AWS_REGION=$SM_REGION ENDPOINT_NAME=$SM_ENDPOINT python3 sm.py destroy )
fi

echo; echo "Sweep (anything listed here is still billing):"
for r in $REGIONS; do
  aws ec2 describe-instances --region "$r" --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`]|[0].Value]' --output text | sed "s/^/  $r ec2 /"
  aws sagemaker list-endpoints --region "$r" --query 'Endpoints[].[EndpointName,EndpointStatus]' --output text | sed "s/^/  $r sagemaker /"
done
echo "(sweep done)"
