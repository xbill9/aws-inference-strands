#!/usr/bin/env bash
# Dress rehearsal from nothing: launch, wait, forward, check, run every live
# command once, then offer to tear it all down. Billing starts at step 2.
#
#   stage/rehearse.sh                # all five: g6 inf2 trn1 g5g sm (~30 min to READY)
#   stage/rehearse.sh g6 sm          # only some (sm alone is ~8 min)
#   stage/rehearse.sh --down         # tear down at the end without asking
#   stage/rehearse.sh --keep         # leave everything up for stage/present.py
#
# Every command's output goes to results/rehearsal-<UTC stamp>/ as well as the screen.
set -uo pipefail
export AWS_PAGER=
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
SM_ENDPOINT=${SM_ENDPOINT:-gemma-4-e2b-strands}
SM_REGION=${SM_REGION:-us-east-2}
READY_TIMEOUT=${READY_TIMEOUT:-2100}    # seconds to wait for instances on SSM and the endpoint
CHECK_WAIT=${CHECK_WAIT:-1500}          # seconds check.py retries for the servers to answer

finish=ask; want=()
for a in "$@"; do
  case $a in
    --down) finish=down ;;
    --keep) finish=keep ;;
    g6|inf2|trn1|g5g|sm) want+=("$a") ;;
    *) echo "unknown argument: $a"; sed -n 2,10p "$0"; exit 2 ;;
  esac
done
[ ${#want[@]} -eq 0 ] && want=(g6 inf2 trn1 g5g sm)

out=$root/results/rehearsal-$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$out"
t0=$(date +%s)
step() { printf '\n== [%3d min] %s\n' $(( ($(date +%s) - t0) / 60 )) "$*"; }
run()  { local f=$1; shift; echo "\$ $*" | tee "$out/$f"; "$@" 2>&1 | tee -a "$out/$f"; return "${PIPESTATUS[0]}"; }

teardown() {
  case $finish in
    keep) echo; echo "Left running (--keep). Next: source stage/demo.env && python3 stage/present.py"
          echo "When done: stage/down.sh"; return ;;
    ask)  if [ -t 0 ]; then read -r -p $'\nTear everything down now? [Y/n] ' yn
          else yn=y; fi
          case $yn in [nN]*) echo "Left running. When done: stage/down.sh"; return ;; esac ;;
  esac
  step "tear down"
  "$here/down.sh" 2>&1 | tee "$out/down.txt"
}
trap 'echo; echo "Interrupted. Instances may be running: stage/down.sh"; exit 130' INT

step "1/6 preflight"
fail=0
aws sts get-caller-identity --query Arn --output text || { echo "AWS session expired: run aws login"; fail=1; }
command -v session-manager-plugin >/dev/null || { echo "session-manager-plugin missing"; fail=1; }
python3 -c "import strands, boto3" || { echo "python: strands or boto3 missing"; fail=1; }
[ $fail -eq 0 ] || exit 1
echo "backends: ${want[*]}   results: ${out#$root/}"

step "2/6 launch (billing starts)"
"$here/up.sh" "${want[@]}" 2>&1 | tee "$here/up.log" "$out/up.txt"

step "3/6 wait for instances on SSM and the endpoint (up to $((READY_TIMEOUT / 60)) min)"
online() {  # backend -> prints Online when its tagged instance is running and on SSM
  local r id
  for r in us-east-2 us-east-1 us-west-2 us-west-1; do
    id=$(aws ec2 describe-instances --region "$r" \
         --filters "Name=tag:Name,Values=strands-demo-$1" Name=instance-state-name,Values=running \
         --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null)
    if [ -n "$id" ] && [ "$id" != None ]; then
      aws ssm describe-instance-information --region "$r" --filters "Key=InstanceIds,Values=$id" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null
      return
    fi
  done
  echo "not-running"
}
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
  line=""; pending=0
  for b in "${want[@]}"; do
    if [ "$b" = sm ]; then
      s=$(aws sagemaker describe-endpoint --region "$SM_REGION" --endpoint-name "$SM_ENDPOINT" \
          --query EndpointStatus --output text 2>/dev/null || echo missing)
      case $s in InService|Failed) ;; *) pending=1 ;; esac
    else
      s=$(online "$b"); [ "$s" = Online ] || pending=1
    fi
    line+="$b=$s  "
  done
  printf '  %s  %s\n' "$(date -u +%H:%M:%SZ)" "$line"
  [ $pending -eq 0 ] && break
  [ "$(date +%s)" -ge $deadline ] && { echo "  timed out; continuing with what is up"; break; }
  sleep 30
done

step "4/6 forward ports and wait for every server to answer"
"$here/forward.sh" 2>&1 | tee "$out/forward.txt"
# shellcheck disable=SC1091
source "$here/demo.env"
run check.txt python3 "$here/check.py" --wait "$CHECK_WAIT"; ready=$?

step "5/6 run every live command once"
cd "$root" || exit 1
for b in bedrock "${want[@]}"; do
  [ "$b" = sm ] && b=sagemaker
  run "ask-$b.txt" python3 stage/ask.py "$b"
done
run demo1-engine.txt python3 demo1_counting.py --runs 3
run demo1-rows.txt   python3 demo1_counting.py --runs 3 --tools rows
run demo2.txt        python3 demo2_orchestrator.py
run bench.txt        python3 bench_chat.py --repeats 3

step "6/6 summary"
grep -h -E '^\s+[0-9:]+Z\s+(PASS|FAIL)' "$out/check.txt" | tail -n $(( ${#want[@]} + 1 ))
for f in "$out"/ask-*.txt "$out"/demo*.txt "$out"/bench.txt; do
  grep -q -E 'Traceback|Error|not configured' "$f" && echo "  PROBLEM in ${f#$root/}"
done
if [ "$ready" -eq 0 ]; then echo "  check.py: READY"; else echo "  check.py: NOT READY (see ${out#$root/}/check.txt)"; fi
echo "  elapsed $(( ($(date +%s) - t0) / 60 )) min; everything is in ${out#$root/}/"

teardown
exit "$ready"
