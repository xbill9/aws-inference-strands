# Gemma 4 on AWS: Demo Setup and Rehearsal

#### What the demo runs on

One Strands agent talks to six backends. Five of them are launched by `stage/up.sh`, so allow 25 minutes after launch before everything answers. Bedrock needs no setup. Launch-to-ready times are from the second recorded run.

| Backend | Hardware | Runtime | Region | Launch to ready |
| --- | --- | --- | --- | --- |
| Bedrock | managed, `us.amazon.nova-micro-v1:0` | Bedrock | us-east-1 | always on |
| SageMaker | `ml.g6.xlarge` (L4) | SageMaker endpoint | us-east-2 | 8.0 min |
| trn1 | `trn1.2xlarge` (Trainium) | `xbill9/gemma4-optb:slim` | us-east-2 | 12.8 min |
| g6 | `g6.xlarge` (L4) | vLLM 0.30 | us-east-2 | 13.9 min |
| inf2 | `inf2.xlarge` (Inferentia2) | `xbill9/gemma4-optb:slim` | us-east-2, falls back to us-west-2 | 17.9 min |
| g5g | `g5g.2xlarge` (Graviton2 + T4G) | patched vLLM 0.27.2rc0 | us-east-1 only | 23.7 min |

The GPU, SageMaker and Neuron backends all serve `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` or its Neuron port. Billing starts when `up.sh` runs and stops at `stage/down.sh`.

Local prerequisites: AWS CLI, the Session Manager plugin, and Python with `strands` and `boto3`.

---

#### Dress Rehearsal, a Day or Two Before

One full run through, about 45 minutes, tested exactly as it will be presented. Run everything from `~/aws-inference-strands`.

`stage/rehearse.sh` does steps 1 to 5 and 7 in one command: it launches, waits for every instance to reach SSM, forwards, runs `check.py --wait`, runs every live command once into `results/rehearsal-<stamp>/`, and asks before tearing down. `stage/rehearse.sh g6 sm` rehearses a subset; `--keep` leaves it up for `present.py`; `--down` tears down without asking.

1. Check the account and the login session: `aws sts get-caller-identity`.
2. Launch the backends: `stage/up.sh 2>&1 | tee stage/up.log`. Every backend should report `launched` or `already up`. For a cheaper partial test, name a subset: `stage/up.sh g6 sm`.
3. Wait about 25 minutes, then open the port forwards: `stage/forward.sh && source stage/demo.env`. Expect five SSM forwards plus SageMaker `InService`.
4. Check every backend: `python3 stage/check.py --wait 1500`. It asks each backend one question and one tool call, and must end `READY`.
5. Walk the talk: `python3 stage/present.py`. Enter runs each slide's command and `r` shows the recorded run.
6. Practise the fallback once: `export BACKENDS=bedrock,g6` leaves backends out, and `r` in `present.py` plays the recorded run.
7. Tear down the same day: `stage/down.sh`, then confirm nothing is left running.

To rehearse the words without launching anything, `python3 stage/present.py --list` prints every cue and command for free.

---

#### On the Day

Start 60 minutes before the talk. Capacity is the slow and uncertain step.

| When | Run | Look for |
| --- | --- | --- |
| T-60 min | `aws sts get-caller-identity` | right account, session lasts past the end of the talk |
| T-60 min | `stage/up.sh 2>&1 \| tee stage/up.log` | every backend `launched` or `already up` |
| T-30 min | `stage/forward.sh && source stage/demo.env` | five forwards plus SageMaker `InService` |
| T-30 min | `python3 stage/check.py --wait 1500` | ends `READY` |
| T-5 min | `stage/forward.sh && source stage/demo.env && python3 stage/check.py` | `READY` again; forwards drop after a long idle |
| T-0 | `python3 stage/present.py` beside the deck, terminal font 20pt or larger | |
| After | `stage/down.sh` | nothing left running |

Keep `stage/up.log` from the day: slide 7 shows it.

---

#### Fallbacks and Points to Watch

Every demo has a recorded run in `results/2026-10-05/`, so a backend that dies on stage costs one sentence and a keypress.

- **A backend is down:** say so, leave it out with `export BACKENDS=...`, and press `r` in `present.py` for the recorded run.
- **Capacity:** on the second run inf2 had no capacity in us-east-2 or us-east-1 and `up.sh` placed it in us-west-2 on its own. g5g has only one region, us-east-1, because its AMI exists only there.
- **Login session:** an expired session drops the SSM forwards in the middle of a demo. Check the expiry at T-60.
- **Idle forwards:** forwards drop after a long idle, which is why T-5 re-runs forward, source and check.
- **Cost:** billing runs from `up.sh` to `down.sh`. Run `down.sh` straight after the talk and after the rehearsal.

---

#### Live Commands

`present.py` runs these in order; they take about 75 seconds together. The full running order is in `docs/DEMO-SCRIPT.md`.

| Slide | Command | Shows | Run time |
| --- | --- | --- | --- |
| 12–19 | `python3 stage/ask.py <backend>` | a short answer with tokens, seconds and tokens per second | 0.9 to 3.0 s each |
| 23 | `python3 demo1_counting.py --runs 3` | the agent calls `count_ids`, the engine counts; answer 8 | 15 s |
| 23 | `python3 demo1_counting.py --runs 3 --tools rows` | the agent gets the raw ids and counts them itself | 18 s |
| 24 | `python3 demo2_orchestrator.py` | a Bedrock agent asks every backend, then quotes a scoreboard sorted in code | 5 s |
| 25 | `python3 bench_chat.py --repeats 3` | one 100-token prompt timed on every backend | 33 s |
