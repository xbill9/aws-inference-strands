"""MCP server for the Strands demo: the six Gemma 4 backends, and the engine that counts.

Modelled on the per-rig devops servers in ~/gemma4-dev (boto3, SSM Run Command, tools
annotated read-only / write / destructive), but one server covers every backend this
demo runs, because the demo is about the backends side by side.

    read-only    list_backends      every demo instance and the SageMaker endpoint, in all US regions
                 backend_health     ask one backend for the capital of France, on the box via SSM
                 backend_logs       tail the serving log on one instance via SSM
                 launch_log         tail the last launch or teardown log
                 ask_backend        one chat request through backends.chat_once, timed here
                 count_ids          the demo 1 engine tool: exact count, min and max for a predicate
    write        launch_backends    stage/up.sh in the background; starts billing
    destructive  teardown           stage/down.sh in the background; terminates everything

Launch and teardown call the stage scripts rather than reimplementing them, so the MCP
path and the documented path launch exactly the same thing.

ask_backend reaches the self-hosted rigs through the SSM forwards in stage/demo.env, which
is read at start-up for any variable the environment does not already set. backend_health
needs no forwards: it runs curl on the instance itself.

    python3 mcp_server.py                  # stdio; .mcp.json registers it for Claude Code
    python3 demo1_counting.py --tools mcp  # demo 1 with count_ids served from here
"""

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backends
import demo1_counting

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # keep the schema listable before boto3 is installed
    boto3 = None

    class BotoCoreError(Exception):
        """Fallback used only when boto3 is absent."""

    class ClientError(Exception):
        """Fallback used only when boto3 is absent."""


NAME = "strands-demo"
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(NAME)

mcp = MCPServer(os.getenv("MCP_SERVER_NAME", NAME))
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

REGIONS = ["us-east-2", "us-east-1", "us-west-2", "us-west-1"]
MANAGED_BY = "strands-demo-run"  # the tag stage/up.sh puts on every instance
REPACK = os.getenv("REPACK", "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4")
SM_ENDPOINT = os.getenv("SM_ENDPOINT", "gemma-4-e2b-strands")
SM_REGION = os.getenv("SM_REGION", "us-east-2")

# What runs on each instance, as stage/up.sh launches it.
ON_BOX = {
    "g6": {"port": 8000, "model": REPACK, "log": "docker logs --tail {n} vllm 2>&1"},
    "g5g": {"port": 8000, "model": "google/gemma-4-E2B-it",
            "log": "journalctl -u vllm.service --no-pager -n {n} 2>&1"},
    "inf2": {"port": 8080, "model": "gemma-4-E2B-it", "log": "docker logs --tail {n} gemma4-neuron 2>&1"},
    "trn1": {"port": 8080, "model": "gemma-4-E2B-it", "log": "docker logs --tail {n} gemma4-neuron 2>&1"},
}
EC2_BACKENDS = list(ON_BOX)
ALL_BACKENDS = ["bedrock", "sagemaker", *EC2_BACKENDS]
HEALTH_QUESTION = "What is the capital of France? Answer in one word."


def _load_demo_env() -> None:
    """Take stage/demo.env's values for anything the environment leaves unset."""
    path = os.path.join(HERE, "stage", "demo.env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            for key, value in re.findall(r"(\w+)=(\S+)", line.removeprefix("export ")):
                default = re.fullmatch(r"\$\{\w+:-(.*)\}", value)
                os.environ.setdefault(key, default.group(1) if default else value)


_load_demo_env()


async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)


def _client(service: str, region: str):
    return boto3.client(service, region_name=region)


def _error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        detail = exc.response.get("Error", {})
        return f"❌ AWS {detail.get('Code', 'error')}: {detail.get('Message', exc)}"
    if isinstance(exc, BotoCoreError):
        return f"❌ AWS client error: {exc}"
    return f"❌ {type(exc).__name__}: {exc}"


async def _region_instances(region: str) -> list[dict]:
    response = await _call(
        _client("ec2", region).describe_instances,
        Filters=[{"Name": "tag:ManagedBy", "Values": [MANAGED_BY]},
                 {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    found = []
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            found.append({"backend": tags.get("Name", "").removeprefix("strands-demo-"), "region": region,
                          "instance_id": inst["InstanceId"], "type": inst["InstanceType"],
                          "state": inst["State"]["Name"], "launched": inst["LaunchTime"].isoformat()})
    return found


async def _instances() -> list[dict]:
    per_region = await asyncio.gather(*[_region_instances(r) for r in REGIONS])
    return [i for found in per_region for i in found]


async def _ssm_ping(region: str, instance_id: str) -> str:
    response = await _call(_client("ssm", region).describe_instance_information,
                           Filters=[{"Key": "InstanceIds", "Values": [instance_id]}])
    info = response.get("InstanceInformationList", [])
    return info[0]["PingStatus"] if info else "not registered"


async def _endpoint_status() -> str:
    try:
        response = await _call(_client("sagemaker", SM_REGION).describe_endpoint, EndpointName=SM_ENDPOINT)
        return response["EndpointStatus"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationException":
            return "not found"
        raise


async def _find(backend: str) -> dict:
    running = [i for i in await _instances() if i["backend"] == backend and i["state"] == "running"]
    if not running:
        raise RuntimeError(f"no running strands-demo-{backend} instance in {', '.join(REGIONS)}")
    return running[0]


async def _ssm(region: str, instance_id: str, command: str, timeout: int = 120) -> str:
    ssm = _client("ssm", region)
    response = await _call(ssm.send_command, InstanceIds=[instance_id], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": [command]}, TimeoutSeconds=max(timeout, 30))
    command_id = response["Command"]["CommandId"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = await _call(ssm.get_command_invocation, CommandId=command_id, InstanceId=instance_id)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "InvocationDoesNotExist":
                await asyncio.sleep(2)
                continue
            raise
        if result["Status"] in {"Success", "Failed", "TimedOut", "Cancelled"}:
            output = (result.get("StandardOutputContent", "") + result.get("StandardErrorContent", "")).strip()
            if result["Status"] != "Success":
                raise RuntimeError(f"SSM {result['Status']}: {output[-1500:]}")
            return output
        await asyncio.sleep(2)
    raise TimeoutError(f"SSM command did not finish in {timeout}s")


def _check_backend(backend: str, allowed: list[str]) -> None:
    if backend not in allowed:
        raise ValueError(f"backend must be one of {', '.join(allowed)}")


@mcp.tool(title="List demo backends", annotations=READ_ONLY)
async def list_backends() -> str:
    """Every backend the demo uses and its state right now.

    Searches all four US regions for instances tagged ManagedBy=strands-demo-run, reports each
    one's SSM ping, the SageMaker endpoint's status, and which backends ask_backend is configured
    for (from stage/demo.env, so stale once the forwards close). Counts are computed here.
    """
    try:
        instances, sm_status = await asyncio.gather(_instances(), _endpoint_status())
        pings = await asyncio.gather(*[_ssm_ping(i["region"], i["instance_id"]) if i["state"] == "running"
                                       else asyncio.sleep(0, result="-") for i in instances])
        rows = [{"backend": "bedrock", "where": "us-east-1 (managed)", "state": "always on", "ssm": "-"},
                {"backend": "sagemaker", "where": f"{SM_REGION} endpoint {SM_ENDPOINT}", "state": sm_status,
                 "ssm": "-"}]
        for inst, ping in zip(instances, pings):
            rows.append({"backend": inst["backend"], "where": f"{inst['region']} {inst['instance_id']} {inst['type']}",
                         "state": inst["state"], "launched": inst["launched"], "ssm": ping})
        seen = {r["backend"] for r in rows}
        rows += [{"backend": b, "where": "-", "state": "not launched", "ssm": "-"}
                 for b in EC2_BACKENDS if b not in seen]
        billing = sum(r["state"] in {"pending", "running", "stopping"} for r in rows if r["backend"] in EC2_BACKENDS)
        billing += sm_status not in {"not found", "Deleting"}
        return json.dumps({"backends": rows, "billing_now": billing,
                           "ec2_running": sum(r["state"] == "running" for r in rows if r["backend"] in EC2_BACKENDS),
                           "sagemaker": sm_status,
                           # from the environment and stage/demo.env; stale once the forwards close
                           "ask_backend_configured": [b.name for b in backends.configured()]}, indent=2)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check one backend's health", annotations=READ_ONLY)
async def backend_health(backend: str) -> str:
    """Ask one backend for the capital of France and pass it only if the answer says Paris.

    For g6, g5g, inf2 and trn1 the request runs on the instance via SSM, so no port forward is
    needed. Bedrock and SageMaker are called directly. A non-empty answer is not a pass: a broken
    deploy on this lineage has answered with degenerate repetition.

    Args:
        backend: one of bedrock, sagemaker, g6, g5g, inf2, trn1
    """
    try:
        _check_backend(backend, ALL_BACKENDS)
        if backend in ("bedrock", "sagemaker"):
            if backend == "sagemaker":
                status = await _endpoint_status()
                if status != "InService":
                    return f"❌ sagemaker: endpoint {SM_ENDPOINT} is {status}"
            b = {x.name: x for x in backends.configured()}.get(backend) or backends.Backend(
                "sagemaker", "sagemaker", "SageMaker endpoint", os.getenv("SM_MODEL", REPACK),
                endpoint=SM_ENDPOINT, region=SM_REGION)
            r = await asyncio.to_thread(backends.chat_once, b, HEALTH_QUESTION, 16)
            text, seconds = r["text"], r["seconds"]
            where = "direct"
        else:
            inst = await _find(backend)
            box = ON_BOX[backend]
            body = json.dumps({"model": box["model"], "max_tokens": 16, "temperature": 0.0,
                               "messages": [{"role": "user", "content": HEALTH_QUESTION}]})
            cmd = (f"curl -s -m 90 -o /tmp/strands-health.json -w '%{{http_code}} %{{time_total}}' "
                   f"-H 'content-type: application/json' -d {shlex.quote(body)} "
                   f"http://localhost:{box['port']}/v1/chat/completions; echo; cat /tmp/strands-health.json")
            out = await _ssm(inst["region"], inst["instance_id"], cmd)
            status_line, _, payload = out.partition("\n")
            code, seconds = status_line.split()[0], float(status_line.split()[1])
            if code != "200":
                return f"❌ {backend}: HTTP {code} from port {box['port']} on {inst['instance_id']}: {payload[:300]}"
            text = json.loads(payload)["choices"][0]["message"]["content"]
            where = f"on {inst['instance_id']} ({inst['region']}) via SSM"
        ok = "paris" in text.lower()
        return f"{'✅' if ok else '❌'} {backend}: answered {text.strip()[:80]!r} in {seconds:.2f}s, {where}"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Tail a backend's serving log", annotations=READ_ONLY)
async def backend_logs(backend: str, tail: int = 60) -> str:
    """Tail the serving log on one instance via SSM: the vLLM or Neuron container, or g5g's
    vllm.service. Falls back to the cloud-init log while the server has not started.

    Args:
        backend: one of g6, g5g, inf2, trn1
        tail: lines to return, 1 to 2000
    """
    try:
        _check_backend(backend, EC2_BACKENDS)
        n = max(1, min(int(tail), 2000))
        inst = await _find(backend)
        cmd = (f"({ON_BOX[backend]['log'].format(n=n)}) || "
               f"(echo '--- serving log not available yet; cloud-init:'; tail -n {n} /var/log/cloud-init-output.log)")
        return f"{backend} {inst['instance_id']} ({inst['region']})\n```\n{await _ssm(inst['region'], inst['instance_id'], cmd)}\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Ask a backend", annotations=READ_ONLY)
async def ask_backend(backend: str, prompt: str, max_tokens: int = 100) -> str:
    """One chat request to one backend, timed here, as stage/ask.py does.

    Tokens per second uses the backend's own completion token count over this server's wall
    clock, so it includes network and prefill. Self-hosted backends need the SSM forwards
    (stage/forward.sh).

    Args:
        backend: one of bedrock, sagemaker, g6, g5g, inf2, trn1
        prompt: the question
        max_tokens: completion limit, 1 to 1024
    """
    try:
        found = {b.name: b for b in backends.configured()}
        if backend not in found:
            return f"❌ {backend} is not configured here; configured: {', '.join(found)}. Run stage/forward.sh."
        r = await asyncio.to_thread(backends.chat_once, found[backend], prompt, max(1, min(int(max_tokens), 1024)))
        return json.dumps(r, indent=2)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Count ids", annotations=READ_ONLY)
def count_ids(op: str, value: int) -> dict:
    """Count the ids matching `id <op> value`, computed by the engine.

    Args:
        op: one of >=, >, <=, <, ==
        value: the integer to compare each id against
    """
    return demo1_counting.count(op, value)


def _background(script: str, args: list[str], log: str) -> str:
    path = os.path.join(HERE, "stage", log)
    with open(path, "w") as out:
        proc = subprocess.Popen([os.path.join(HERE, "stage", script), *args], cwd=HERE, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    return f"started stage/{script} {' '.join(args)} (pid {proc.pid}); output in stage/{log}, read it with launch_log"


@mcp.tool(title="Launch demo backends", annotations=WRITE)
async def launch_backends(names: list[str] | None = None) -> str:
    """Run stage/up.sh in the background. Billing starts now and runs until teardown.

    A backend already pending or running is left alone. Ready in 8 to 24 minutes depending on
    the backend; poll list_backends and backend_health.

    Args:
        names: any of g6, inf2, trn1, g5g, sm; empty launches all five
    """
    try:
        names = names or []
        bad = [n for n in names if n not in {"g6", "inf2", "trn1", "g5g", "sm"}]
        if bad:
            return f"❌ unknown backend(s) {bad}; use g6 inf2 trn1 g5g sm"
        return _background("up.sh", names, "up.log")
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Tear down every demo backend", annotations=DESTRUCTIVE)
async def teardown() -> str:
    """Run stage/down.sh in the background: terminate every instance tagged
    ManagedBy=strands-demo-run in all US regions, destroy the SageMaker endpoint, close the
    forwards, then sweep for anything still billing. Read the sweep with launch_log("down")."""
    try:
        return _background("down.sh", [], "down.log")
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Read the launch or teardown log", annotations=READ_ONLY)
def launch_log(which: str = "up", tail: int = 60) -> str:
    """Tail stage/up.log or stage/down.log.

    Args:
        which: up or down
        tail: lines to return
    """
    if which not in ("up", "down"):
        return "❌ which must be up or down"
    path = os.path.join(HERE, "stage", f"{which}.log")
    if not os.path.exists(path):
        return f"stage/{which}.log does not exist yet"
    with open(path) as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-max(1, min(int(tail), 2000)):])


if __name__ == "__main__":
    mcp.run()
