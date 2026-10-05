# Demo Script: Gemma 4 Inference on AWS

The running order for the 60-minute AWS Community Builders talk. It covers the deck's 30 slides, a live moment in each section, and what to do when a backend is down.

Deck: https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM

---

#### Before the Talk

| When | Do | Check |
|---|---|---|
| T-60 min | `cd ~/aws-inference-strands && stage/up.sh 2>&1 \| tee stage/up.log` | every backend says `launched` or `already up` |
| T-60 min | `aws sts get-caller-identity` | the right account, a session that lasts past the talk |
| T-30 min | `stage/forward.sh && source stage/demo.env` | five `localhost:… ->` lines plus `sm: … InService` |
| T-30 min | `python3 stage/check.py --wait 1500` | ends with `READY` |
| T-5 min | `stage/forward.sh && source stage/demo.env && python3 stage/check.py` | `READY` again; forwards drop after a long idle |
| T-5 min | terminal font at 20pt or more, one pane, `clear` | |
| T-0 | `python3 stage/present.py` beside the deck | one cue per slide; Enter runs that slide's command, `r` shows its recorded run |

Measured launch to ready in the second run (`results/2026-10-05-stage/REPORT.md`): SageMaker 8.0 min, trn1 12.8 min, g6 13.9 min, inf2 17.9 min, g5g 23.7 min. The g5g's first start reads the weights from a fresh EBS volume. `up.sh` tries every zone and then the next region when one has no capacity. Where each backend landed is in `stage/up.log`, which you show on slide 7.

**Fallback.** Every demo has a recorded run in `results/2026-10-05/`. If a backend fails on stage, say so, `BACKENDS=` it out of the next command, and `cat` the recorded run.

```bash
export BACKENDS=bedrock,g6,sagemaker,g5g    # leave out what is down
cat results/2026-10-05/demo1-engine.txt
```

---

#### Running Order

`stage/present.py` steps through this table one slide at a time: talking points, the slide's link, the minutes planned against the clock, and the live command, with the recorded run one key away. `python3 stage/present.py --list` prints it all without running anything.

| Slides | Section | Minutes | Live |
|---|---|---|---|
| 1–3 | Title, About, Where to Find Me | 3 | |
| 4–5 | One question, six ways; the six backends | 4 | |
| 6–10 | Strategies | 8 | `cat stage/up.log` on slide 7 |
| 11–13 | Managed: Bedrock, SageMaker | 6 | `ask.py bedrock`, `ask.py sagemaker` |
| 14–16 | GPUs: g6, g5g | 6 | `ask.py g6`, `ask.py g5g` |
| 17–20 | Neuron: Inferentia2, Trainium, vLLM-Neuron | 8 | `ask.py inf2`, `ask.py trn1` |
| 21–26 | Strands: model classes, demo 1, demo 2, bench, cost | 15 | four commands, about 75 s of runtime in all |
| 27–30 | So, Which One?; clean up; links; questions | 10 | `stage/down.sh` after the talk |

That is 60 minutes with questions at the end. If you run long, cut the `ask.py` calls on slides 13 and 16: demo 2 hits every backend anyway.

---

#### Slide 7 — Quota Does Not Mean Availability

```bash
cat stage/up.log
```

The log shows each zone that refused a launch and the region the instance landed in. On 2026-10-05 inf2 had quota in us-east-2 and no capacity in any of its zones, nor in us-east-1; it launched in us-west-2. Point at those lines.

---

#### Slides 12–19 — One Question per Backend

```bash
python3 stage/ask.py bedrock
python3 stage/ask.py sagemaker
python3 stage/ask.py g6
python3 stage/ask.py g5g
python3 stage/ask.py inf2
python3 stage/ask.py trn1
```

Each writes a short paragraph and prints the token count, the seconds and the tokens per second, timed in the script: 0.9 to 3.0 s per command in the rehearsal. Pass a question as the second argument to ask something else. Say what serves each one: a managed API; the AWS vLLM container; vLLM 0.30 on an L4; a patched vLLM on Graviton2 and a T4G; the hand-ported image on Inferentia2; the same image on Trainium without a rebuild.

---

#### Slide 23 — Demo 1: Same Agent, Every Backend That Calls Tools

```bash
python3 demo1_counting.py --runs 3          # 15 s
```

Eleven ids, "how many are 10 or more?", answer 8. The agent gets `count_ids(op, value)` and the engine does the counting. Each run is graded twice: did the answer say 8, and did the filter it sent select the same ids as `id >= 10`. The recorded run had 20 of 20 on both for all four backends.

Then the contrast:

```bash
python3 demo1_counting.py --runs 3 --tools rows   # 18 s
```

Here the agent gets the raw ids and counts them itself. The point for the room: push the arithmetic into the tool, then check the filter the model sent, because a wrong filter returns an exact, wrong number.

---

#### Slide 24 — Demo 2: Right Table, Wrong Summary

```bash
python3 demo2_orchestrator.py                # 5 s
```

A Bedrock agent asks every backend through `ask_<name>` tools, then quotes `scoreboard()`, which sorts the times in code. The table is right. Read the agent's sentence under it: in one recorded run it named Trainium the fastest while the table it had just quoted put g6 first. Trainium does win now and then on this one-word question: in one of five rehearsal runs it beat g6 by 9 ms. The last line, `[harness] … not asked by the agent`, comes from code and lists any backend the agent skipped.

---

#### Slide 25 — Same Prompt, Every Backend

```bash
python3 bench_chat.py --repeats 3            # 33 s
```

One 100-token prompt, timed in code. Recorded medians, first run then rehearsal: g6 131.5 and 132.5 tok/s, Bedrock 111.9 and 115.9, SageMaker 106.3 and 105.2, then g5g, trn1 and inf2 all between 35.8 and 37.2. The three groups hold from run to run; the order inside the last group swaps.

---

#### After the Talk

```bash
stage/down.sh
```

It closes the forwards, terminates every instance tagged `ManagedBy=strands-demo-run` in the four US regions, deletes the SageMaker endpoint, its config and model, then lists anything still running in those regions. The security groups and the instance profile stay for the next run.
