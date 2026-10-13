---
title: "One MCP Server for Six AWS Inference Backends: Bedrock, SageMaker, GPUs, Inferentia and Trainium"
published: false
description: "A step by step look at the MCP server that runs a six-backend Gemma 4 demo on AWS: one Python file over the demo's own code, eight tools marked read-only, write or destructive, SSM Run Command instead of open ports, and an agent that counts through the server and gets the same grade as in-process tools."
tags: aws, mcp, python, ai
cover_image: https://raw.githubusercontent.com/xbill9/aws-inference-strands/main/docs/article/mcp/devto-cover.75e9e51b.jpg
---

This article provides a step by step guide to the MCP server that operates a six-backend inference demo on AWS: Amazon Bedrock, a SageMaker endpoint, vLLM on EC2 g6 and g5g, and Gemma 4 on Inferentia2 and Trainium. A suite of eight Python MCP tools lists, launches, health-checks, reads the logs of, asks, and tears down every backend from one chat session.

https://github.com/xbill9/aws-inference-strands

---

#### What Is the Problem?

The demo behind this server puts one Strands agent in front of six ways to serve a model on AWS. Running it means five billable resources in up to four regions, two of them on AWS Neuron chips, and none of the instances reachable from the internet.

The repository already has shell scripts for all of it: `stage/up.sh`, `stage/forward.sh`, `stage/check.py`, `stage/ask.py` and `stage/down.sh`. Each one prints for a person at a terminal.

The MCP server gives the same operations to an agent. It answers "what is running and what is billing", "is inf2 serving yet", "show me the vLLM log on g6" and "tear it all down" with tools whose arithmetic and checks run in code, and it serves demo 1's counting tool to a Strands agent over the protocol itself.

---

#### At This Point You Should Have…

- An AWS account with the AWS CLI configured, and Bedrock model access for Amazon Nova Micro in us-east-1
- Python 3 (3.14.7 here) with `mcp`, `strands-agents[openai,sagemaker]` and `boto3` installed
- The instance profile and SSM-only security group `stage/up.sh` uses (it creates the security group on first use)
- Claude Code, or any MCP client that can start a stdio server
- The [companion article](https://dev.to/aws-builders/gemma-4-inference-on-aws-bedrock-sagemaker-gpus-inferentia-and-trainium-behind-one-strands-agent-1dm0) for how each backend is built

---

#### Step 1 — Register the Server

The server is one file, `mcp_server.py`, started over stdio. `.mcp.json` in the repository root registers it for Claude Code:

```json
{
  "mcpServers": {
    "strands-demo": {
      "command": "python3",
      "args": ["mcp_server.py"]
    }
  }
}
```

Any MCP client can start it the same way. Three lines of JSON-RPC on stdin are enough to see what it offers:

```shell
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"printf","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | python3 mcp_server.py
```

```json
{
  "protocolVersion": "2025-11-25",
  "capabilities": {"tools": {"listChanged": false}, "prompts": {"listChanged": false},
                   "resources": {"listChanged": false, "subscribe": false}, "experimental": {}},
  "serverInfo": {"name": "strands-demo", "version": ""}
}
```

The server runs on the MCP Python SDK 2.1.1 and its `MCPServer` class, and negotiates protocol version `2025-11-25`. `tools/list` returns eight tools.

---

#### Step 2 — One File Over the Demo's Own Code

The server imports the two modules the demos already use and adds AWS calls around them.

```text
           Claude Code  /  Strands MCPClient
                     │  stdio, JSON-RPC 2.0
                     ▼
 ┌──────────────────────── mcp_server.py ────────────────────────┐
 │ list_backends  backend_health  backend_logs   launch_log      │
 │ ask_backend    count_ids       launch_backends  teardown      │
 └──┬───────────────┬───────────────┬───────────────┬────────────┘
    │ boto3         │ backends.py   │ demo1_        │ subprocess
    │ EC2, SSM,     │ chat_once()   │ counting.py   │ stage/up.sh
    │ SageMaker     │               │ count()       │ stage/down.sh
    ▼               ▼               ▼               ▼
    4 US regions,   Bedrock,        the 11 ids,     detached,
    by tag          SageMaker,      in memory       output in
                    SSM forwards                    stage/*.log
```

Four design rules hold the file together:

- **Reuse, then wrap.** `ask_backend` calls `backends.chat_once`, the function `stage/ask.py`, `stage/check.py`, the benchmark and demo 2 call. `count_ids` calls `demo1_counting.count`, the function the in-process tool calls. A number from the server and a number from the scripts come from the same line of code.
- **Launch and teardown run the stage scripts.** `launch_backends` starts `stage/up.sh` and `teardown` starts `stage/down.sh`, so the MCP route and the documented route create and destroy the same resources.
- **No inbound ports.** Every instance sits in a security group with no inbound rules. Tools that need the box run a shell command on it through SSM Run Command.
- **The tag is the inventory.** `stage/up.sh` tags every instance `ManagedBy=strands-demo-run`, and every lookup in the server filters on that tag across `us-east-2`, `us-east-1`, `us-west-2` and `us-west-1`. A backend that fell over to another region on a capacity error is still found.

boto3 is synchronous, so each call runs in a worker thread through `asyncio.to_thread`. That lets one tool ask four regions at once with `asyncio.gather`:

```python
async def _instances() -> list[dict]:
    per_region = await asyncio.gather(*[_region_instances(r) for r in REGIONS])
    return [i for found in per_region for i in found]
```

---

#### Step 3 — Read-Only, Write and Destructive

Every tool carries MCP tool annotations, so a client can tell an inventory call from one that bills or one that deletes before it calls anything.

```python
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)
```

What `tools/list` returns, read back through the MCP client:

| Tool | Annotation | Arguments | What it touches |
|---|---|---|---|
| `list_backends` | read-only | none | EC2, SSM and SageMaker describe calls in 4 regions |
| `backend_health` | read-only | `backend` | one chat request, on the box via SSM for EC2 |
| `backend_logs` | read-only | `backend`, `tail` | the serving log, via SSM |
| `ask_backend` | read-only | `backend`, `prompt`, `max_tokens` | one chat request through the forwards |
| `count_ids` | read-only | `op`, `value` | demo 1's ids, in memory |
| `launch_log` | read-only | `which`, `tail` | `stage/up.log` or `stage/down.log` |
| `launch_backends` | write | `names` | `stage/up.sh`; starts billing |
| `teardown` | destructive | none | `stage/down.sh`; terminates everything |

The rest of this article takes the tools in the order a demo day uses them.

---

#### Step 4 — `list_backends`: What Is Running and What Is Billing

`list_backends` searches the four regions for tagged instances, asks SSM whether each running one has checked in, asks SageMaker for the endpoint's status, and adds up what is billing. The counts are computed in the tool, so the agent quotes them.

With nothing launched:

```json
{
  "backends": [
    {"backend": "bedrock", "where": "us-east-1 (managed)", "state": "always on", "ssm": "-"},
    {"backend": "sagemaker", "where": "us-east-2 endpoint gemma-4-e2b-strands", "state": "not found", "ssm": "-"},
    {"backend": "g6", "where": "-", "state": "not launched", "ssm": "-"},
    {"backend": "g5g", "where": "-", "state": "not launched", "ssm": "-"},
    {"backend": "inf2", "where": "-", "state": "not launched", "ssm": "-"},
    {"backend": "trn1", "where": "-", "state": "not launched", "ssm": "-"}
  ],
  "billing_now": 0,
  "ec2_running": 0,
  "sagemaker": "not found",
  "ask_backend_configured": ["bedrock", "g6", "g5g", "sagemaker", "inf2", "trn1"]
}
```

One call took 0.78 seconds for five AWS describe calls: one per region for EC2 and one for SageMaker. Each running instance adds an SSM call.

`ask_backend_configured` lists all six backends while five are down. It names every backend with a URL in `demo.env`, the file `stage/forward.sh` writes into the `stage` directory, and that file outlives the forwards. The tool's description says so, and Step 8 shows what a call to one of those backends returns.

---

#### Step 5 — `launch_backends` and `launch_log`: Start Billing, Then Read the Launch

`launch_backends` takes any of `g6`, `inf2`, `trn1`, `g5g` and `sm`, checks the names, and starts `stage/up.sh` with them in the background:

```plaintext
launch_backends {"names": ["g6", "inf2"]}
started stage/up.sh g6 inf2 (pid 2362067); output in stage/up.log, read it with launch_log
```

The call returns at once. The script picks an AMI, tries each zone in turn, then the next region, and writes what it did to `stage/up.log`. `launch_log` reads the log:

```plaintext
strands-demo-g6: launched i-005ebaee08f191e7a	us-east-2a
strands-demo-inf2: no capacity in us-east-2 (subnet-0b0f0d29473b0a624), trying the next zone
strands-demo-inf2: launched i-0bf43d9b77b321ec3	us-east-2b

Next: stage/forward.sh (once the instances are up), then: source stage/demo.env && python3 stage/check.py
```

inf2 had no capacity in the first zone and launched in the second. Half a minute later `list_backends` showed both instances running and billing, before either had registered with SSM:

```json
{"backend": "inf2", "where": "us-east-2 i-0bf43d9b77b321ec3 inf2.xlarge", "state": "running", "ssm": "not registered"},
{"backend": "g6",   "where": "us-east-2 i-005ebaee08f191e7a g6.xlarge",   "state": "running", "ssm": "not registered"},
...
"billing_now": 2,
"ec2_running": 2
```

---

---

#### Step 6 — `backend_health`: Ask on the Box

For the four EC2 backends, `backend_health` finds the instance by tag and sends one shell command through SSM Run Command. The command runs `curl` against the server's port on the instance and prints the HTTP code, the time and the body:

```python
cmd = (f"curl -s -m 90 -o /tmp/strands-health.json -w '%{{http_code}} %{{time_total}}' "
       f"-H 'content-type: application/json' -d {shlex.quote(body)} "
       f"http://localhost:{box['port']}/v1/chat/completions; echo; cat /tmp/strands-health.json")
```

The request never leaves the instance, so the check needs no port forward and no inbound rule. Bedrock and SageMaker are called directly. SageMaker is checked for `InService` first.

Polled once a minute from launch, the replies trace each instance coming up:

```plaintext
02:13:14Z g6   -> ❌ AWS InvalidInstanceId: Instances not in a valid state for account
02:14:21Z g6   -> ❌ RuntimeError: SSM Failed: 000 0.000140
...
02:24:36Z g6   -> ✅ g6: answered 'Paris' in 1.21s, on i-005ebaee08f191e7a (us-east-2) via SSM
02:26:11Z inf2 -> ❌ RuntimeError: SSM Failed: 000 90.000889
...
02:32:44Z inf2 -> ✅ inf2: answered 'Paris' in 0.27s, on i-0bf43d9b77b321ec3 (us-east-2) via SSM
```

Each failure names a different stage:

| Reply | Meaning |
|---|---|
| `InvalidInstanceId` | the SSM agent has not registered yet; nothing can run on the box |
| `000` in under a millisecond | the box answers, the serving port is closed |
| `000` after 90.0 s | the port is open and the request waited out `curl -m 90` |
| ✅ `'Paris'` | the model is serving and answers correctly |

The 90-second replies came from inf2 only. Its server opens port 8080 while it still loads the compiled model onto the NeuronCores, and holds requests until it is ready.

g6 first passed 12.2 minutes after the launch call and inf2 20.4 minutes after it, with the checks a minute apart. g6's first answer took 1.21 seconds and the next three took 0.02 to 0.03 seconds.

---

---

#### Step 7 — `backend_logs`: The Serving Log Without a Shell

`backend_logs` runs the right log command for each backend through SSM: `docker logs` for the vLLM container on g6 and the Neuron container on inf2 and trn1, `journalctl -u vllm.service` on g5g. Until the container exists it falls back to the cloud-init log, which shows the image pull:

```plaintext
g6 i-005ebaee08f191e7a (us-east-2)
Error response from daemon: No such container: vllm
--- serving log not available yet; cloud-init:
...
469e56fbf4cf: Pull complete
```

Two minutes after g6's first health check passed, the same call returned vLLM's own log, with the health check's request in it:

```plaintext
(APIServer pid=1) INFO:     Application startup complete.
(EngineCore pid=108) WARNING 10-13 02:24:34 [jit_monitor.py:141] Triton kernel JIT compilation during inference: kernel_unified_attention. This causes a latency spike; consider extending warmup to cover this shape/config.
(APIServer pid=1) INFO:     172.17.0.1:34422 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

The JIT warning matches the 1.21-second first answer. On inf2 the log showed the load in progress, then the result:

```plaintext
HTTP listening on :8080 — /health=503 'loading' until ready; queue max=8, gen timeout 120.0s
...
READY in 473.8s — serving on :8080 (SLIM host, peak RSS 14.45 GB)
[req] pt=21 ct=1 5.8tok/s 0.17s finish=stop
```

The first line explains the 90-second health check replies: the port opened before the model was ready.

---

---

#### Step 8 — `ask_backend`: One Timed Request

`ask_backend` sends one chat request through `backends.chat_once` and returns the text, the backend's own completion token count, and the time measured in the server. Tokens per second is that count over that time, so it includes the network and the prompt processing. Self-hosted backends are reached through the SSM port forwards `stage/forward.sh` opens.

Before the forwards, with `demo.env` left over from an earlier session:

```plaintext
ask_backend {"backend": "g6", "prompt": "hello"}
❌ URLError: <urlopen error [Errno 111] Connection refused>
```

After `stage/forward.sh` rewrote `demo.env`, a new server process read it at start-up:

```json
{"backend": "g6",   "label": "vLLM on EC2 g6 (L4)",            "text": "Hello! How can I help you today? 😊",
 "completion_tokens": 11, "seconds": 0.14,  "tokens_per_second": 78.6}
{"backend": "inf2", "label": "Inferentia2 (inf2, hand-ported)", "text": "Hello! How can I help you today?",
 "completion_tokens": 9,  "seconds": 0.423, "tokens_per_second": 21.3}
```

```plaintext
ask_backend {"backend": "trn1", "prompt": "hello"}
❌ trn1 is not configured here; configured: bedrock, g6, inf2. Run stage/forward.sh.
```

`list_backends` now reported `ask_backend_configured: ["bedrock", "g6", "inf2"]` and both instances `Online` in SSM. Replies this short are mostly prompt processing and network, so these rates describe one request each; the [companion article](https://dev.to/aws-builders/gemma-4-inference-on-aws-bedrock-sagemaker-gpus-inferentia-and-trainium-behind-one-strands-agent-1dm0) has the 100-token benchmark.

---

#### Step 9 — `count_ids`: Demo 1's Engine Over MCP

Demo 1 asks how many of eleven ids are 10 or more. The answer is 8. `count_ids` takes a predicate and returns the exact count, minimum and maximum, and echoes the predicate it ran:

```plaintext
count_ids {"op": ">=", "value": 10}  ->  {"where": "id >= 10", "count": 8, "min": 10, "max": 23}
count_ids {"op": ">",  "value": 10}  ->  {"where": "id > 10",  "count": 7, "min": 11, "max": 23}
count_ids {"op": "=>", "value": 10}  ->  {"error": "op must be one of ['<', '<=', '==', '>', '>=']"}
```

The second line returns an exact, sourced 7 for a predicate the question did not ask for. The `where` field is in the response so the predicate can be checked along with the number.

`demo1_counting.py --tools mcp` offers this tool to a Strands agent through the protocol. Strands' `MCPClient` starts the server over stdio and a tool filter exposes `count_ids` alone:

```python
MCPClient(lambda: stdio_client(StdioServerParameters(command=sys.executable, args=[server])),
          tool_filters={"allowed": ["count_ids"]})
```

The tool runs in the server's process, so the demo cannot record calls as they happen. It reads them back from the agent's conversation instead, and grades them with the same function the in-process run uses:

```python
def calls_from(messages: list) -> list:
    return [{"tool": c["toolUse"]["name"], **c["toolUse"].get("input", {})}
            for m in messages if m["role"] == "assistant" for c in m["content"] if "toolUse" in c]
```

Twenty runs each on Bedrock, the in-process tool and then the MCP tool:

```shell
BACKENDS=bedrock python3 demo1_counting.py --tools engine --runs 20
BACKENDS=bedrock python3 demo1_counting.py --tools mcp --runs 20
```

```plaintext
True answer: 8  tools: engine
...
backend     runs  correct  right-filter  errors  median s  filters sent
bedrock       20   20/20          20/20       0      1.07  id >= 10

True answer: 8  tools: mcp
...
backend     runs  correct  right-filter  errors  median s  filters sent
bedrock       20   20/20          20/20       0      1.07  id >= 10
```

Same answers, same filter, same median time. Starting the server costs once per demo run, outside the timed agent calls.

---

#### Step 10 — `teardown`: Stop Everything That Bills

`teardown` takes no arguments and starts `stage/down.sh` in the background. It is the one tool marked destructive. The script closes the forwards, terminates every instance tagged `ManagedBy=strands-demo-run` in all four regions, deletes the SageMaker endpoint when it exists, and then lists anything still running in any of the four regions.

```plaintext
teardown {}
started stage/down.sh  (pid 2367066); output in stage/down.log, read it with launch_log
```

Twenty-one seconds later, `launch_log` with `which` set to `down`:

```plaintext
us-east-2: terminating i-0bf43d9b77b321ec3	i-005ebaee08f191e7a

Sweep (anything listed here is still billing):
(sweep done)
```

and `list_backends`:

```plaintext
"billing_now": 0,
"ec2_running": 0,
"sagemaker": "not found"
```

The sweep and the inventory both read empty while the two instances were still in `shutting-down`, because both look only at pending, running, stopping and stopped instances.

---

#### 🔎 Tip: Return the Error as Text the Agent Can Read

Every tool catches its exceptions and returns one line that starts with ❌ and names the cause: an AWS error code and message, a `ValueError` for a bad backend name, a `RuntimeError` naming the four regions searched.

```plaintext
backend_health {"backend": "tpu"}  ->  ❌ ValueError: backend must be one of bedrock, sagemaker, g6, g5g, inf2, trn1
launch_backends {"names": ["p5"]}  ->  ❌ unknown backend(s) ['p5']; use g6 inf2 trn1 g5g sm
```

An agent reads the cause in the reply. The protocol-level `isError` flag stays `false` on all of them, so a client that branches on that flag treats a failed health check as a success. Set it where clients act on it.

---

#### 🔎 Tip: Pass a Health Check on the Answer

`backend_health` asks "What is the capital of France? Answer in one word." at temperature 0 with 16 tokens, and passes only when the reply contains "paris". A non-empty reply proves the port is open. A broken build of this model family has answered with repeated text, and that reply is non-empty too.

---

#### 🔎 Tip: Clamp Every Number the Model Sends

`tail` is clamped to 1 to 2000 lines and `max_tokens` to 1 to 1024. `backend` is checked against a fixed list before any AWS call, and `launch_backends` rejects an unknown name before `stage/up.sh` starts. A model that asks for a million log lines gets two thousand.

---

#### 🔎 Tip: Run Long Jobs Detached and Hand Back a Log

`stage/up.sh` and `stage/down.sh` each took under half a minute here, and the backends behind them take up to twenty minutes to serve. Both scripts run with `start_new_session=True` and their output in `stage/up.log` or `stage/down.log`, so the tool call returns at once and the job survives the MCP session closing. `launch_log` reads the log back.

---

#### Compare and Contrast

| | stage scripts | MCP server |
|---|---|---|
| Caller | a person at a terminal | any MCP client: Claude Code, a Strands agent |
| Launch and teardown | `stage/up.sh`, `stage/down.sh` | the same scripts, detached, read back with `launch_log` |
| Inventory | `aws ec2 describe-instances` per region | 🥇 one call, four regions in parallel, billing count computed |
| Health | `stage/check.py`, needs the forwards | 🥇 on the box via SSM, no forwards |
| Logs | `aws ssm start-session`, then `docker logs` | 🥇 `backend_logs`, cloud-init fallback |
| One question | `stage/ask.py`, needs the forwards | `ask_backend`, needs the forwards |
| Safety signal | the script's name | read-only, write, destructive annotations |

---

#### So, Which One?

🥇 **The MCP server** to operate the demo from a chat session: inventory, health, logs and teardown in plain questions, with the billing count and the pass or fail computed in code.

🥈 **The stage scripts** for the talk itself and anything scripted, and for the port forwards, which `ask_backend` and the demos need and the server does not open.

Both launch and stop the same resources, because one calls the other.

---

#### Summary

The goal of this article was to operate a six-backend AWS inference demo through MCP. The key to the solution was one server over the demo's own code, reaching instances through SSM Run Command and launching through the same scripts a person runs. The results were:

- 🟢 **Eight tools in one file over the demo's own code**: the same `chat_once` and `count` functions the scripts call, and the same `stage/up.sh` and `stage/down.sh`.
- 🟢 **Every tool is marked read-only, write or destructive**, and `tools/list` carries the marks to the client.
- 🟢 **Health checks and logs run on the instance through SSM Run Command**, with no inbound rules and no port forwards; g6 passed 12.2 minutes after launch and inf2 20.4 minutes after.
- 🟢 **A launch, health checks, logs, requests and a teardown ran from one session**, and `list_backends` read `billing_now: 0` 21 seconds after `teardown`.
- 🟢 **Demo 1 over MCP matched demo 1 in-process on Bedrock**: 20 of 20 correct with the right filter both ways, median 1.07 seconds both ways.
- ⚠️ **Failed calls return a ❌ line with `isError` false**, so a client that branches on the protocol flag treats them as successes.
- ⚠️ **`ask_backend` depends on `demo.env`**, which outlives the forwards; it lists backends that are down until `stage/forward.sh` runs again.
- ❌ **The server opens no port forwards**, so `ask_backend` and the demos still need `stage/forward.sh` from a terminal.

Scope: one session on 2026-10-13 (UTC), us-east-2, two on-demand instances (g6.xlarge and inf2.xlarge) from launch to teardown, with Bedrock in us-east-1. trn1, g5g and the SageMaker endpoint were not launched, so their tools are shown only in their not-running replies. Health checks were one minute apart, so each time to ready is accurate to about a minute. Demo 1 ran 20 times per tool set on Bedrock only, with the MCP Python SDK 2.1.1, Strands Agents 1.55.0 and Python 3.14.7 on a laptop.

The strategy for using MCP to operate Gemma 4 inference backends across AWS was validated with an incremental step by step approach.

---

#### References

* [aws-inference-strands | GitHub](https://github.com/xbill9/aws-inference-strands)
* [Gemma 4 Inference on AWS: Bedrock, SageMaker, GPUs, Inferentia and Trainium Behind One Strands Agent | DEV](https://dev.to/aws-builders/gemma-4-inference-on-aws-bedrock-sagemaker-gpus-inferentia-and-trainium-behind-one-strands-agent-1dm0)
* [Model Context Protocol specification, tools | modelcontextprotocol.io](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
* [MCP Python SDK | GitHub](https://github.com/modelcontextprotocol/python-sdk)
* [Strands Agents, MCP tools | strandsagents.com](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/mcp-tools/)
* [AWS Systems Manager Run Command | AWS](https://docs.aws.amazon.com/systems-manager/latest/userguide/run-command.html)
