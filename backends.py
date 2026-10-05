"""The inference backends the two demos talk to, and how to reach each one.

Every backend is configured from the environment, so the same code runs against
whatever is up on the day. A backend whose variable is unset is skipped, not
faked: the demos print which ones they could reach.

    kind      what Strands uses                    tool calls
    bedrock   strands.models.BedrockModel          yes
    openai    strands.models.openai.OpenAIModel    yes, when vLLM runs with --enable-auto-tool-choice
    sagemaker strands.models.sagemaker.SageMakerAIModel   yes, same vLLM flags via SM_VLLM_*
    neuron    plain HTTP to the hand-ported Gemma 4 server  no: tools are ignored and the
              prompt bucket is 128 tokens, so it is only ever called as a tool (demo 2)

The self-hosted rigs accept no inbound traffic; reach them through an SSM port
forward (see README.md) and point the *_URL variables at localhost.
"""

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class Backend:
    name: str
    kind: str
    label: str
    model: str
    url: Optional[str] = None
    endpoint: Optional[str] = None
    region: Optional[str] = None

    @property
    def tool_capable(self) -> bool:
        return self.kind != "neuron"


def configured() -> list[Backend]:
    """Every backend whose settings are present in the environment."""
    env = os.environ.get
    found = [
        Backend("bedrock", "bedrock", "Amazon Bedrock",
                env("BEDROCK_MODEL", "us.amazon.nova-micro-v1:0"),
                region=env("BEDROCK_REGION", "us-east-1")),
    ]
    if env("G6_URL"):
        found.append(Backend("g6", "openai", "vLLM on EC2 g6 (L4)",
                             env("G6_MODEL", "google/gemma-4-E2B-it"), url=env("G6_URL")))
    if env("G5G_URL"):
        found.append(Backend("g5g", "openai", "vLLM on EC2 g5g (Graviton2 + T4G)",
                             env("G5G_MODEL", "google/gemma-4-E2B-it"), url=env("G5G_URL")))
    if env("SM_ENDPOINT"):
        found.append(Backend("sagemaker", "sagemaker", "SageMaker endpoint (vLLM, L4)",
                             env("SM_MODEL", "google/gemma-4-E2B-it"),
                             endpoint=env("SM_ENDPOINT"), region=env("SM_REGION", "us-east-2")))
    if env("INF2_URL"):
        found.append(Backend("inf2", "neuron", "Inferentia2 (inf2, hand-ported)",
                             env("INF2_MODEL", "gemma-4-E2B-it"), url=env("INF2_URL")))
    if env("TRN1_URL"):
        found.append(Backend("trn1", "neuron", "Trainium (trn1, hand-ported)",
                             env("TRN1_MODEL", "gemma-4-E2B-it"), url=env("TRN1_URL")))
    only = env("BACKENDS")
    if only:
        keep = {b.strip() for b in only.split(",")}
        found = [b for b in found if b.name in keep]
    return found


def _tolerate_vllm_usage(sm) -> None:
    """Let strands.models.sagemaker read vLLM 0.30's usage block.

    Strands 1.55 builds its UsageMetadata dataclass with **usage, and that class
    declares four fields. vLLM 0.30 also sends completion_tokens_details, so every
    SageMaker call raised TypeError before the answer was read. Drop the fields
    the class does not declare; the token counts themselves are unchanged.
    """
    original = sm.UsageMetadata
    if getattr(original, "_tolerant", False):
        return
    known = set(original.__dataclass_fields__)

    def tolerant(**usage):
        return original(**{k: v for k, v in usage.items() if k in known})

    tolerant._tolerant = True
    sm.UsageMetadata = tolerant


def strands_model(b: Backend, temperature: float = 0.0, max_tokens: int = 512):
    """The Strands model object for a tool-capable backend."""
    if b.kind == "bedrock":
        from strands.models import BedrockModel
        return BedrockModel(model_id=b.model, region_name=b.region,
                            temperature=temperature, max_tokens=max_tokens)
    if b.kind == "openai":
        from strands.models.openai import OpenAIModel
        return OpenAIModel(client_args={"base_url": b.url, "api_key": "unused"}, model_id=b.model,
                           params={"temperature": temperature, "max_tokens": max_tokens})
    if b.kind == "sagemaker":
        import strands.models.sagemaker as sm
        from strands.models.sagemaker import SageMakerAIModel
        _tolerate_vllm_usage(sm)
        return SageMakerAIModel(
            endpoint_config={"endpoint_name": b.endpoint, "region_name": b.region},
            payload_config={"max_tokens": max_tokens, "stream": False, "temperature": temperature})
    raise ValueError(f"{b.name} ({b.kind}) cannot drive a Strands agent: it has no tool calls")


def chat_once(b: Backend, prompt: str, max_tokens: int = 64) -> dict:
    """One plain chat request, timed here. Returns text, tokens and seconds.

    Tokens per second is computed from the backend's own completion token count
    and this function's wall clock, so it includes network and prefill.
    """
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    if b.kind == "bedrock":
        import boto3
        r = boto3.client("bedrock-runtime", region_name=b.region).converse(
            modelId=b.model, messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0})
        text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
        completion = r["usage"]["outputTokens"]
    elif b.kind == "sagemaker":
        import boto3
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        r = boto3.client("sagemaker-runtime", region_name=b.region).invoke_endpoint(
            EndpointName=b.endpoint, ContentType="application/json", Body=json.dumps(body))
        out = json.loads(r["Body"].read())
        text = out["choices"][0]["message"]["content"]
        completion = out["usage"]["completion_tokens"]
    else:
        body = {"model": b.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        req = urllib.request.Request(b.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read())
        text = out["choices"][0]["message"]["content"]
        completion = out["usage"]["completion_tokens"]
    seconds = time.perf_counter() - t0
    return {"backend": b.name, "label": b.label, "text": text.strip(), "completion_tokens": completion,
            "seconds": round(seconds, 3), "tokens_per_second": round(completion / seconds, 1) if seconds else 0.0}
