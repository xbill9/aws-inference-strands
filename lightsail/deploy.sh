#!/usr/bin/env bash
# EmbeddingGemma 2 (int8/int4 text-only repack) on a Lightsail container service, CPU only.
# Builds the CPU image from ~/embedding-cloudrun, pushes it to Lightsail and deploys it on a
# public HTTPS endpoint. Power "medium" is 1 vCPU / 2 GB, the same shape as emb-cpu-repack on
# Cloud Run. Billed per month while the service exists (lightsail/down.sh deletes it).
#   lightsail/deploy.sh            then   python3 ~/embedding-cloudrun/scripts/bench.py lightsail-medium URL
set -euo pipefail
SERVICE=${SERVICE:-emb-cpu-repack}
POWER=${POWER:-medium}
REGION=${REGION:-us-east-1}
SRC=${SRC:-$HOME/embedding-cloudrun}
IMAGE=embeddinggemma2-cpu-repack:lightsail

docker build --build-arg MODEL=repack --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t "$IMAGE" "$SRC"
aws lightsail get-container-services --service-name "$SERVICE" --region "$REGION" >/dev/null 2>&1 \
  || aws lightsail create-container-service --service-name "$SERVICE" --power "$POWER" --scale 1 \
       --region "$REGION" --query 'containerService.state' --output text
until [ "$(aws lightsail get-container-services --service-name "$SERVICE" --region "$REGION" \
           --query 'containerServices[0].state' --output text)" != PENDING ]; do sleep 10; done
ref=$(aws lightsail push-container-image --service-name "$SERVICE" --label repack --image "$IMAGE" \
        --region "$REGION" | grep -o ':'"$SERVICE"'\.repack\.[0-9]*' | tail -1)
aws lightsail create-container-service-deployment --service-name "$SERVICE" --region "$REGION" \
  --containers "{\"app\":{\"image\":\"$ref\",\"environment\":{\"TORCH_THREADS\":\"1\"},\"ports\":{\"8080\":\"HTTP\"}}}" \
  --public-endpoint '{"containerName":"app","containerPort":8080,"healthCheck":{"path":"/health","intervalSeconds":30,"timeoutSeconds":10,"healthyThreshold":2,"unhealthyThreshold":5}}' \
  --query 'containerService.state' --output text
until [ "$(aws lightsail get-container-services --service-name "$SERVICE" --region "$REGION" \
           --query 'containerServices[0].state' --output text)" = RUNNING ]; do sleep 15; done
aws lightsail get-container-services --service-name "$SERVICE" --region "$REGION" \
  --query 'containerServices[0].url' --output text
