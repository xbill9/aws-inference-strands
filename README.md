# One Strands agent, every AWS inference path

Two demos for a survey of LLM inference on AWS: Amazon Bedrock, a SageMaker endpoint, vLLM on an EC2 GPU, and hand-ported Gemma 4 on AWS Inferentia2 and Trainium.

| Demo | What it shows | Backends |
|---|---|---|
| `demo1_counting.py` | The same tool-using agent and question on each backend, graded on the answer **and** on the filter the model sent | Bedrock, vLLM on g6 and g5g, SageMaker |
| `demo2_orchestrator.py` | A Bedrock agent that calls every backend as a tool and quotes a scoreboard computed in code | all six |
| `mcp_server.py` | An MCP server over the same backends: list, health check, logs, ask, launch, tear down, and demo 1's `count_ids` engine | all six |
| `bench_chat.py` | The same 100-token prompt on every backend, timed in code | all six |
| `cost.py` | USD per hour and per million output tokens, from AWS Pricing API list prices and measured throughput; write-up in [docs/article/devto-aws-inference-cost.md](docs/article/devto-aws-inference-cost.md) | all six |

`lightsail/` deploys the CPU-only EmbeddingGemma 2 repack from `~/embedding-cloudrun` on a Lightsail container service.

The Neuron builds take part in demo 2 only: the hand-ported server ignores tool definitions and its prompt bucket holds 128 tokens, which a Strands agent's system prompt and tool schemas do not fit.

## Bring the backends up

`stage/` launches, reaches, checks and tears down every self-hosted backend. The talk's running order, with a live command for each section, is in [docs/DEMO-SCRIPT.md](docs/DEMO-SCRIPT.md). A one-page version for the day is [docs/CHEATSHEET.md](docs/CHEATSHEET.md). The slides are [on Google Slides](https://docs.google.com/presentation/d/1Ax8KGF8T0EYffshY7WfUM61QcHmLk0q72EE7G_hIwgM/edit?usp=sharing).

```bash
stage/up.sh 2>&1 | tee stage/up.log      # g6, inf2, trn1, g5g, SageMaker; starts billing
stage/forward.sh && source stage/demo.env # SSM port forwards, writes the variables below
python3 stage/check.py --wait 1500        # one question and one tool call per backend; READY
python3 stage/ask.py trn1                 # one backend, one question, timed
python3 stage/present.py                  # step through the talk, slide by slide
stage/down.sh                             # terminate, delete the endpoint, sweep four regions
```

`up.sh` tries each zone, then the next region, when a launch gets no capacity. It reuses an SSM-only security group in each region and creates one with no inbound rules where none exists.

## Install

```bash
pip install 'strands-agents[openai,sagemaker]' boto3
```

## Configure

Each backend is turned on by its variable; unset means skipped. `BACKENDS=bedrock,g6` limits a run to some of them.

| Variable | Backend | Example |
|---|---|---|
| `BEDROCK_MODEL`, `BEDROCK_REGION` | Bedrock (always on) | `us.amazon.nova-micro-v1:0`, `us-east-1` |
| `G6_URL`, `G6_MODEL` | vLLM on EC2 g6 | `http://localhost:8000/v1` |
| `G5G_URL`, `G5G_MODEL` | vLLM on EC2 g5g (Graviton2 + T4G, patched vLLM 0.27.2rc0) | `http://localhost:8004/v1` |
| `SM_ENDPOINT`, `SM_REGION` | SageMaker real-time endpoint | `gemma-4-e2b`, `us-east-2` |
| `INF2_URL` | Gemma 4 on inf2 (`xbill9/gemma4-optb:slim`) | `http://localhost:8081/v1` |
| `TRN1_URL` | the same image on trn1 | `http://localhost:8082/v1` |

The EC2 rigs take no inbound traffic. Forward a local port to each through SSM:

```bash
aws ssm start-session --region us-east-2 --target <instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8082"]}'
```

### Tool calls on the GPU backends

vLLM returns tool calls only when started with them enabled:

- **g6 rig** (`~/gemma4-dev/gpu-vllm-g6-2b`): `EXTRA_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser gemma4"`
- **SageMaker** (`~/sagemaker-gemma`): `SM_VLLM_ENABLE_AUTO_TOOL_CHOICE=true SM_VLLM_TOOL_CALL_PARSER=gemma4 make deploy`

## Run

```bash
python3 demo1_counting.py                       # engine counts; 5 runs per backend
python3 demo1_counting.py --tools rows          # the model counts the rows itself
python3 demo1_counting.py --tools mcp           # count_ids served by mcp_server.py over MCP
python3 demo1_counting.py --runs 20 --json results/demo1.json
python3 demo2_orchestrator.py
```

## MCP server

`mcp_server.py` follows the per-rig servers in `~/gemma4-dev`: boto3 and SSM Run Command, no inbound ports, every tool annotated read-only, write or destructive. `.mcp.json` registers it for Claude Code as `strands-demo`; any MCP client can start it with `python3 mcp_server.py`.

| Tool | Kind | What it does |
|---|---|---|
| `list_backends` | read-only | every instance tagged `ManagedBy=strands-demo-run` in the four US regions, its SSM ping, the SageMaker endpoint status, and how many are billing |
| `backend_health` | read-only | asks one backend for the capital of France and passes only on Paris; the EC2 backends are asked on the box through SSM, so no forward is needed |
| `backend_logs` | read-only | tails the vLLM or Neuron container log, or g5g's `vllm.service`, falling back to cloud-init |
| `ask_backend` | read-only | one timed chat request, as `stage/ask.py`; uses the forwards in `stage/demo.env` |
| `count_ids` | read-only | demo 1's engine: exact count, minimum and maximum for `id <op> value` |
| `launch_log` | read-only | tails `stage/up.log` or `stage/down.log` |
| `launch_backends` | write | runs `stage/up.sh` in the background; billing starts |
| `teardown` | destructive | runs `stage/down.sh` in the background |

Launch and teardown run the stage scripts, so both routes start and stop the same resources. With `--tools mcp`, demo 1 offers the agent only `count_ids` from this server and grades the calls it reads back from the agent's messages, so it is graded exactly as `--tools engine`.

## How the grading works

Demo 1 asks how many of eleven ids are 10 or more; the answer is 8. With `--tools engine` the agent gets `count_ids(op, value)`, which returns the exact count, minimum and maximum. A run counts as correct when the final answer states 8, and as the **right filter** only when one of its `count_ids` calls selects the same ids as `id >= 10`. A run that sends `id > 10` gets 7 back from the engine; a model that then says 8 is right by accident, and the table shows it.

Demo 2 times every request in code and reads each backend's own completion token count, so tokens per second includes the network and the prefill. `scoreboard()` sorts and formats the table; the agent is told to quote it as is, and the harness prints which backends the agent never asked.
