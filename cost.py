"""Cost of each backend, from AWS list prices and measured throughput. All arithmetic is here.

    python3 cost.py results/rehearsal-20261012T1429Z/bench.txt --out results/2026-10-12-cost

Prices come from the AWS Pricing API at run time (On-Demand, Linux, shared tenancy) and are
written to OUT/prices.json. Throughput comes from two places:

- one request at a time: the median tok/s of every backend in a bench_chat.py table
- 1, 4 and 16 requests in parallel: OUT/parallel-g6.json and OUT/parallel-sagemaker.json, the
  compare.py runs of google/gemma-4-E2B-it-qat-w4a16-ct on a g6.xlarge (2026-09-29) and an
  ml.g6.xlarge endpoint (2026-09-25), both vLLM 0.30.0, client on the instance for the g6

USD per million output tokens = USD per hour / (tok/s * 3600) * 1e6, so it assumes the instance
is busy for the whole hour. Bedrock is billed per token and has no hourly price.
"""

import argparse
import datetime
import json
import os
import re

import boto3

HOURS_PER_MONTH = 730
# backend -> (service, instance type, region); g5g is us-east-1 only, the rest ran in us-east-2
INSTANCES = {
    "g6": ("ec2", "g6.xlarge", "us-east-2"),
    "g5g": ("ec2", "g5g.2xlarge", "us-east-1"),
    "inf2": ("ec2", "inf2.xlarge", "us-east-2"),
    "trn1": ("ec2", "trn1.2xlarge", "us-east-2"),
    "sagemaker": ("sagemaker", "ml.g6.xlarge", "us-east-2"),
}
BEDROCK = ("Nova Micro", "us-east-1")


def on_demand(products):
    """First On-Demand price per unit in a get-products result."""
    for s in products["PriceList"]:
        for term in json.loads(s)["terms"].get("OnDemand", {}).values():
            for dim in term["priceDimensions"].values():
                return float(dim["pricePerUnit"]["USD"])
    raise LookupError("no On-Demand price")


def fetch_prices():
    p = boto3.client("pricing", region_name="us-east-1")
    eq = lambda k, v: {"Type": "TERM_MATCH", "Field": k, "Value": v}
    out = {"fetched": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "usd_per_hour": {}}
    for name, (svc, itype, region) in INSTANCES.items():
        if svc == "ec2":
            f = [eq("instanceType", itype), eq("regionCode", region), eq("operatingSystem", "Linux"),
                 eq("tenancy", "Shared"), eq("preInstalledSw", "NA"), eq("capacitystatus", "Used")]
            out["usd_per_hour"][name] = on_demand(p.get_products(ServiceCode="AmazonEC2", Filters=f))
        else:
            f = [eq("instanceName", itype), eq("regionCode", region), eq("component", "Hosting")]
            out["usd_per_hour"][name] = on_demand(p.get_products(ServiceCode="AmazonSageMaker", Filters=f))
    model, region = BEDROCK
    prefix = "USE1-" + model.replace(" ", "")
    for kind in ("input", "output"):
        f = [eq("regionCode", region), eq("usagetype", f"{prefix}-{kind}-tokens")]
        out[f"bedrock_usd_per_1k_{kind}"] = on_demand(p.get_products(ServiceCode="AmazonBedrock", Filters=f))
    return out


def read_bench(path):
    """bench_chat.py table -> {backend: median tok/s}."""
    rates = {}
    for line in open(path):
        m = re.match(r"(\w+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+-[\d.]+\s+\d+-\d+", line)
        if m:
            rates[m.group(1)] = float(m.group(2))
    return rates


def per_million(usd_hour, tok_s):
    return usd_hour / (tok_s * 3600) * 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prices = fetch_prices()
    json.dump(prices, open(os.path.join(a.out, "prices.json"), "w"), indent=2)
    hourly, rates = prices["usd_per_hour"], read_bench(a.bench)
    br_out = prices["bedrock_usd_per_1k_output"] * 1000

    lines = [f"prices fetched {prices['fetched']}; bench {a.bench}", "",
             "One request at a time (bench_chat.py, 100-token answer)", "",
             f"{'backend':10s} {'tok/s':>7s} {'USD/hr':>7s} {'USD/month 24x7':>15s} {'USD / 1M output tokens':>23s}"]
    rows = sorted(rates.items(), key=lambda kv: per_million(hourly[kv[0]], kv[1]) if kv[0] in hourly else br_out)
    for name, tok_s in rows:
        if name == "bedrock":
            lines.append(f"{name:10s} {tok_s:7.1f} {'-':>7s} {'0 idle':>15s} {br_out:23.2f}  (per-token list price, "
                         f"input {prices['bedrock_usd_per_1k_input'] * 1000:.3f})")
        else:
            h = hourly[name]
            lines.append(f"{name:10s} {tok_s:7.1f} {h:7.4f} {h * HOURS_PER_MONTH:15.0f} {per_million(h, tok_s):23.2f}")

    lines += ["", "Requests in parallel (compare.py, 256-token answers, google/gemma-4-E2B-it-qat-w4a16-ct)", "",
              f"{'backend':10s} {'parallel':>8s} {'tok/s':>8s} {'USD / 1M output tokens':>23s}"]
    for name in ("g6", "sagemaker"):
        load = json.load(open(os.path.join(a.out, f"parallel-{name}.json")))["load"]
        for n, v in sorted(load.items(), key=lambda kv: int(kv[0])):
            t = v["aggregate_tokens_per_second_median"]
            lines.append(f"{name:10s} {n:>8s} {t:8.1f} {per_million(hourly[name], t):23.3f}")

    g6 = hourly["g6"] * HOURS_PER_MONTH
    lines += ["", f"g6 24x7 = USD {g6:.0f}/month. Bedrock spend matches it at:"]
    for ratio in (0, 5):  # input tokens per output token in the request mix
        blended = br_out + ratio * prices["bedrock_usd_per_1k_input"] * 1000
        m_tokens = g6 / blended
        lines.append(f"  {ratio} input tokens per output token: USD {blended:.3f} per 1M output tokens, "
                     f"{m_tokens:.0f}M output tokens/month = {m_tokens * 1e6 / (HOURS_PER_MONTH * 3600):.0f} "
                     f"tok/s around the clock")
    text = "\n".join(lines)
    open(os.path.join(a.out, "cost.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
