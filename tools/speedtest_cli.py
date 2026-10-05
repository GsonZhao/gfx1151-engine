#!/usr/bin/env python3
"""Headless runner for llm_speedtest (python/llm_test_backend.py), same metric
as the halogen screenshot: concurrency 1, random-word prompts, stream=True,
prefill = prompt_tokens / (server prompt time, else TTFT - net latency).

Usage:
  PYTHONPATH=<llm_speedtest>/python python3 tools/speedtest_cli.py \
      --url http://127.0.0.1:8080/v1/chat/completions --model x \
      --lengths 4096,8192,16384,32768 --out 64
Needs: httpx, fastapi, pydantic, slowapi (imports of the backend module).
"""
import argparse, asyncio, json, sys

import llm_test_backend as B  # noqa: E402

# halogen table (2026-10-05 screenshot, AI MAX 395, V2 weights)
HALOGEN = {4096: 1653.81, 8192: 1852.05, 16384: 1869.53, 32768: 1838.78,
           65536: 1767.39, 131072: 1674.77, 262144: 1543.39}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--key", default="")
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=120000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",") if x]

    cal = await B.send_warmup_request(a.url, a.key, "openai", a.model, a.timeout, 0.7, 0.9)
    lat = []
    for _ in range(2):
        await B.collect_network_latency_sample(lat, a.url, a.key, "openai", a.model, a.timeout, 0.7, 0.9)

    rows = []
    for L in lengths:
        nl = await B.collect_network_latency_sample(lat, a.url, a.key, "openai", a.model, a.timeout, 0.7, 0.9)
        est = B.estimate_generated_prompt_tokens(L, calibration=cal)
        to = B.calculate_dynamic_timeout(est, a.timeout)
        r = await B.execute_single_request(
            api_url=a.url, api_key=a.key, api_type="openai", model_name=a.model,
            prompt_length=L, output_length=a.out, timeout=to, temperature=0.7, top_p=0.9,
            presence_penalty=0.0, frequency_penalty=0.0, seed=1, network_latency_ms=nl,
            network_latency_sample_count=len(lat), prompt_calibration=cal)
        if not r.get("success"):
            rows.append((L, None, None, None, None, r.get("error")))
            continue
        rows.append((L, r["prompt_tokens"], r["ttft_ms"], r["prefill_speed"],
                     r["output_speed"], r["prefill_time_source"]))
        await asyncio.sleep(1.5)

    print("\n==== speedtest (llm_speedtest metric, concurrency 1) ====", file=sys.stderr)
    print(f"{'len':>7} {'ptok':>7} {'TTFT ms':>10} {'prefill':>9} {'halogen':>8} {'ratio':>6} {'decode':>7}  src",
          file=sys.stderr)
    out = []
    for L, pt, tt, pf, dc, src in rows:
        h = HALOGEN.get(L)
        if pf is None:
            print(f"{L:>7} FAILED {src}", file=sys.stderr)
            continue
        ratio = f"{pf / h:.3f}" if h else "-"
        print(f"{L:>7} {pt:>7} {tt:>10.1f} {pf:>9.1f} {h or 0:>8.1f} {ratio:>6} {dc:>7.1f}  {src}",
              file=sys.stderr)
        out.append(dict(len=L, prompt_tokens=pt, ttft_ms=tt, prefill=pf, decode=dc, src=src))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
