# Demo Cheat Sheet

Run everything from `~/aws-inference-strands`. Deck: https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit

All five self-hosted backends cost about $4.59 an hour together, from `up.sh` until `down.sh`.

---

#### Before the Talk

| When | Run | Expect |
|---|---|---|
| T-90 | `aws sts get-caller-identity` | right account; session lasts past the end of the talk |
| T-90 | `stage/up.sh 2>&1 \| tee stage/up.log` | each backend `launched` or `already up`; a full zone moves to the next one by itself |
| T-60 | `stage/forward.sh && source stage/demo.env` | five port forwards plus SageMaker `InService` |
| T-60 | `python3 stage/check.py --wait 1500` | ends `READY` |
| T-5 | `stage/forward.sh && source stage/demo.env && python3 stage/check.py` | `READY` again; forwards drop when idle |
| T-0 | `python3 stage/present.py` | terminal font 20 pt or larger, next to the deck |

Ready after launch in the 2026-10-12 rehearsal: Bedrock and SageMaker 9 min, trn1 15, g6 and inf2 16, g5g 24.

---

#### During the Talk

`present.py` keys (type, then Enter):

| Key | Does |
|---|---|
| Enter | run this slide's next command, or go to the next slide |
| `n` / `p` | next / previous slide |
| `g 29` | go to slide 29 |
| `r` | show the recorded run instead of running live |
| `s` | skip this command |
| `l` | print this slide's deck link |
| `q` | quit |

Restart part way: `python3 stage/present.py --start 27`. Print every cue without running: `python3 stage/present.py --list`.

---

#### Live Commands

| Slide | Command | Takes |
|---|---|---|
| 9 | `grep -E 'no capacity\|launched\|already up' stage/up.log` | instant |
| 14 | `python3 stage/ask.py bedrock` | ~1 s |
| 16 | `python3 stage/ask.py sagemaker` | ~1 s |
| 18 | `python3 stage/ask.py g6` | ~1 s |
| 20 | `python3 stage/ask.py g5g` | ~3 s |
| 23 | `python3 stage/ask.py inf2` | ~3 s |
| 24 | `python3 stage/ask.py trn1` | ~3 s |
| 29 | `python3 demo1_counting.py --runs 3` | ~15 s |
| 29 | `python3 demo1_counting.py --runs 3 --tools rows` | ~18 s |
| 30 | `python3 demo2_orchestrator.py` | ~5 s |
| 31 | `python3 bench_chat.py --repeats 3` | ~33 s |

Short on time: skip slides 16 and 20 with `s`; demo 2 on slide 30 calls every backend anyway.

---

#### When Something Breaks

| Problem | Do |
|---|---|
| a backend does not answer | `export BACKENDS=bedrock,g6,sagemaker` (leave it out), then `r` for its recorded run |
| everything local fails | `aws sts get-caller-identity`; if expired, log in again, then `stage/forward.sh && source stage/demo.env` |
| forwards dropped | `stage/forward.sh && source stage/demo.env` |
| demo 2 summary comes out right | press `r`: slide 30 is about the recorded run where the summary was wrong |
| inf2 has no capacity in us-east-2 | `up.sh` places it in us-west-2 on its own; wait for it |

Recorded runs: `results/2026-10-05-stage/` (what `r` plays) and `results/rehearsal-20261012T1429Z/`.

---

#### After the Talk

```
stage/down.sh
```

Terminates the instances, deletes the SageMaker endpoint, config and model, and sweeps four regions. The last line lists anything still billing; it should be empty.
