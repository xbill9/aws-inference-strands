#!/usr/bin/env bash
# One SageMaker endpoint at a time (us-east-2 quota is 1 x ml.g6.xlarge).
#   run.sh stock gemma-4-e2b-stock google/gemma-4-E2B-it
#   run.sh repack gemma-4-e2b-strands xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
set -euo pipefail
tag=$1; export SM_ENDPOINT=$2 SM_MODEL=$3 SM_REGION=us-east-2 BACKENDS=sagemaker
d=$(dirname "$0"); cd "$d/../.."
python3 bench_chat.py --repeats 10 --json "$d/bench-$tag.json"        | tee "$d/bench-$tag.txt"
python3 demo1_counting.py --runs 20 --json "$d/demo1-engine-$tag.json" | tee "$d/demo1-engine-$tag.txt"
python3 demo1_counting.py --runs 20 --tools rows --json "$d/demo1-rows-$tag.json" | tee "$d/demo1-rows-$tag.txt"
