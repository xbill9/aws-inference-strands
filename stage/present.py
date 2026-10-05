"""Step through the talk: one cue per slide, with the live command for that slide.

Run it in the terminal you will show, next to the Google Slides deck:

    source stage/demo.env
    python3 stage/present.py              # from slide 1
    python3 stage/present.py --start 21   # pick up at the Strands section
    python3 stage/present.py --list       # print every cue and command, run nothing

Keys (type the letter, then Enter):
    Enter   run this slide's next command; with none left, go to the next slide
    n / p   next / previous slide          g N   go to slide N
    r       show the recorded run for the current command instead of running it
    s       skip the current command       l     show the slide's link again
    q       quit

Each slide header shows its link (#slide=id.sl_NN) so the deck can be jumped to
from the terminal, the minutes planned so far, and the clock since slide 1. A
command whose backend is not configured, or that exits non-zero, offers its
recorded run from results/2026-10-05-stage/.
"""

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DECK = "https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit"
RECORDED = "results/2026-10-05-stage"
BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def ask(backend: str) -> dict:
    return {"cmd": f"python3 stage/ask.py {backend}", "needs": backend,
            "recorded": f"{RECORDED}/ask-{backend}.txt"}


# (slide, title, minutes, notes, commands)
CUES = [
    (1, "Gemma 4 Inference on AWS", 1.0,
     ["Six ways to serve one model on AWS, one Strands agent over all of them."], []),
    (2, "About me", 1.0, ["GDE Cloud and AI/ML, AWS Community Builder."], []),
    (3, "Where to find me", 0.5, ["Everything today is on GitHub and dev.to."], []),
    (4, "One Question, Six Ways to Answer It", 2.0,
     ["Managed API, managed endpoint, your GPU, your AWS chip.",
      "Grading and timing happen in code, never in the model."], []),
    (5, "The Six Backends", 2.0,
     ["g6 and SageMaker: same weights, same GPU type; VM against managed endpoint.",
      "inf2 and trn1: same image digest, two chips."], []),
    (6, "Strategies", 0.5, ["Four habits that decided whether a run happened at all."], []),
    (7, "Quota Does Not Mean Availability", 2.5,
     ["Rehearsal: inf2 refused in five zones across us-east-2 and us-east-1, launched in us-west-2a.",
      "Quota lets you ask; the zone decides. Keep a second zone and a second region."],
     [{"cmd": "grep -E 'no capacity|launched|already up' stage/up.log", "recorded": f"{RECORDED}/up.log"}]),
    (8, "Debug With the Small Model", 2.0,
     ["E2B on Neuron is ready in about 2 minutes, the 26B in close to 4.",
      "Wiring breaks the same way at any size; the small model breaks faster."], []),
    (9, "Check Your Numbers, Verify Results", 2.0,
     ["Code counts, times and ranks; the model quotes.",
      "Grade the filter the agent sent, as well as the number it said."], []),
    (10, "Keep IAM Loose While You Build", 1.5,
     ["One instance profile, one SSM-only security group per region, no inbound ports.",
      "Tighten before anything faces users."], []),
    (11, "Managed: Bedrock and SageMaker", 0.5, [], []),
    (12, "Bedrock: Nothing to Start", 2.5,
     ["An IAM permission and an inference profile id.",
      "At temperature 0 Bedrock gave 9 different paragraphs in 10 repeats."],
     [ask("bedrock")]),
    (13, "SageMaker: The Same vLLM, Managed", 3.0,
     ["8.0 and 8.2 minutes to InService over two runs.",
      "Same text, word for word, as the g6 on slide 15."],
     [ask("sagemaker")]),
    (14, "GPUs on EC2", 0.5, [], []),
    (15, "g6.xlarge: vLLM 0.30 on an NVIDIA L4", 3.0,
     ["About 14 minutes launch to ready, both runs.",
      "Fastest decode here: 131.5 then 132.45 tok/s. Compare its paragraph with SageMaker's."],
     [ask("g6")]),
    (16, "g5g.2xlarge: Arm Host, NVIDIA T4G", 2.5,
     ["Patched vLLM for sm_75; weights read from a fresh EBS volume for about 530 s.",
      "23.7 minutes launch to ready: start it first."],
     [ask("g5g")]),
    (17, "AWS Silicon: Neuron", 0.5, [], []),
    (18, "Inferentia2: Gemma 4, Ported by Hand", 3.0,
     ["No vLLM release serves Gemma 4 on Neuron, so the model is compiled by hand.",
      "This one is running in us-west-2 if the rehearsal failover repeated."],
     [ask("inf2")]),
    (19, "Trainium: The Same Image, No Rebuild", 3.0,
     ["Same digest as inf2; the paragraph comes back word for word the same.",
      "36.9 against inf2's 36.6 tok/s."],
     [ask("trn1")]),
    (20, "vLLM on Neuron Lags the GPU Releases", 1.5,
     ["The newest vLLM-Neuron lists Trn2 and Trn3 only; Trn1/Inf2 stop at vLLM 0.16."], []),
    (21, "One Strands Agent", 0.5, [], []),
    (22, "One Agent, Four Model Classes", 2.0,
     ["BedrockModel, OpenAIModel with a base_url, SageMakerAIModel.",
      "Neuron backends are called as tools: no tool calls, 128-token prompt."], []),
    (23, "Demo 1: Every Backend Picked the Right Filter", 4.0,
     ["Eleven ids; how many are 10 or more? The answer is 8.",
      "count_ids does the counting. The grade checks the answer and the filter sent.",
      "Then the contrast: --tools rows hands the model the ids to count itself."],
     [{"cmd": "python3 demo1_counting.py --runs 3", "recorded": f"{RECORDED}/stage-demo1-engine.txt"},
      {"cmd": "python3 demo1_counting.py --runs 3 --tools rows", "recorded": f"{RECORDED}/stage-demo1-rows.txt"}]),
    (24, "Demo 2: The Table Was Right, the Sentence Was Wrong", 3.5,
     ["A Bedrock agent asks every backend; scoreboard() sorts in code.",
      "Read the table, then the agent's sentence. trn1 beat g6 in one of five rehearsal runs.",
      "The [harness] line lists any backend the agent skipped."],
     [{"cmd": "python3 demo2_orchestrator.py", "recorded": f"{RECORDED}/stage-demo2.txt"}]),
    (25, "Same Prompt: vLLM on g6 Decoded Fastest", 3.0,
     ["100 tokens, timed in code, network and prefill included.",
      "Three groups that hold from run to run: g6; Bedrock and SageMaker; g5g, trn1, inf2."],
     [{"cmd": "python3 bench_chat.py --repeats 3", "recorded": f"{RECORDED}/stage-bench.txt"}]),
    (26, "Per Token, the g6 Costs Least", 2.0,
     ["Arithmetic: hourly price over measured tokens per hour, one request at a time."], []),
    (27, "So, Which One?", 2.5,
     ["Bedrock first; g6 for your own weights; SageMaker to manage it; Neuron when the model exists for it."], []),
    (28, "Clean Up Four Regions", 1.5,
     ["stage/down.sh terminates, deletes the endpoint and sweeps four regions.",
      "Run it after the talk, not now."], []),
    (29, "Links", 1.0, ["Repo, rigs, write-up."], []),
    (30, "Questions", 5.0, [], []),
]


def configured() -> set:
    try:
        import backends
        return {b.name for b in backends.configured()}
    except Exception:
        return set()


def show_file(path: str) -> None:
    full = os.path.join(REPO, path)
    print(f"{YELLOW}[recorded run: {path}]{RESET}")
    try:
        with open(full) as f:
            sys.stdout.write(f.read())
    except OSError as exc:
        print(f"{RED}cannot read it: {exc}{RESET}")


def run(c: dict, have: set) -> bool:
    need = c.get("needs")
    if need and need not in have:
        print(f"{YELLOW}{need} is not configured (source stage/demo.env?). r shows the recorded run.{RESET}")
        return False
    print(f"{BOLD}$ {c['cmd']}{RESET}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(c["cmd"], shell=True, cwd=REPO)
    took = time.perf_counter() - t0
    if rc != 0:
        print(f"{RED}exit {rc} after {took:.1f} s. r shows the recorded run.{RESET}")
        return False
    print(f"{DIM}({took:.1f} s){RESET}")
    return True


def header(i: int, start: float) -> None:
    n, title, minutes, notes, cmds = CUES[i]
    planned = sum(c[2] for c in CUES[:i])
    elapsed = (time.time() - start) / 60
    drift = elapsed - planned
    pace = (f"{GREEN}on time{RESET}" if abs(drift) < 1 else
            f"{YELLOW}{drift:+.1f} min{RESET}" if abs(drift) < 3 else f"{RED}{drift:+.1f} min{RESET}")
    print("\n" + "─" * 72)
    print(f"{BOLD}Slide {n}/{len(CUES)}  {title}{RESET}   {DIM}{minutes:g} min{RESET}")
    print(f"{DIM}planned {planned:.1f} min · clock {elapsed:.1f} min · {RESET}{pace}")
    print(f"{DIM}{DECK}#slide=id.sl_{n:02d}{RESET}")
    for line in notes:
        print(f"  • {line}")
    for c in cmds:
        print(f"  {BOLD}▶ {c['cmd']}{RESET}")


def list_cues() -> None:
    total = sum(c[2] for c in CUES)
    for n, title, minutes, notes, cmds in CUES:
        print(f"{n:>2}  {minutes:>4g} min  {title}")
        for c in cmds:
            print(f"              $ {c['cmd']}")
    print(f"\n{len(CUES)} slides, {total:g} minutes planned, "
          f"{sum(len(c[4]) for c in CUES)} live commands")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1, help="slide to start on")
    ap.add_argument("--list", action="store_true", help="print the cues and commands, run nothing")
    a = ap.parse_args()
    if a.list:
        list_cues()
        return 0

    have = configured()
    print(f"Backends configured: {', '.join(sorted(have)) or 'none (source stage/demo.env)'}")
    i = max(0, min(len(CUES) - 1, a.start - 1))
    # Starting mid-deck counts as being on schedule at that slide.
    start = time.time() - 60 * sum(c[2] for c in CUES[:i])
    step = 0
    header(i, start)
    while True:
        cmds = CUES[i][4]
        cur = cmds[step] if step < len(cmds) else None
        prompt = f"[Enter: run {cur['cmd'].removeprefix('python3 ')}]" if cur else "[Enter: next slide]"
        try:
            key = input(f"{DIM}{prompt} n p g N r s l q > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if key == "q":
            return 0
        if key == "" and cur:
            run(cur, have)
            step += 1
            continue
        if key == "r" and cur:
            show_file(cur["recorded"])
            step += 1
            continue
        if key == "r" and step > 0 and cmds:
            show_file(cmds[step - 1]["recorded"])
            continue
        if key == "s" and cur:
            step += 1
            continue
        if key == "l":
            print(f"{DECK}#slide=id.sl_{CUES[i][0]:02d}")
            continue
        if key in ("", "n"):
            if i == len(CUES) - 1:
                print("Last slide. q to quit.")
                continue
            i, step = i + 1, 0
        elif key == "p":
            i, step = max(0, i - 1), 0
        elif key.startswith("g"):
            try:
                i, step = max(0, min(len(CUES) - 1, int(key[1:].strip()) - 1)), 0
            except ValueError:
                print("g needs a slide number, e.g. g 23")
                continue
        else:
            print("Keys: Enter n p g N r s l q")
            continue
        header(i, start)


if __name__ == "__main__":
    sys.exit(main())
