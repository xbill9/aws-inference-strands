#!/usr/bin/env bash
# Delete the Lightsail container service lightsail/deploy.sh created (stops its monthly billing).
set -euo pipefail
aws lightsail delete-container-service --service-name "${SERVICE:-emb-cpu-repack}" --region "${REGION:-us-east-1}"
