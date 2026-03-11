import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

URL = "http://127.0.0.1:8000/generate/stream"

PAYLOAD = {
    "prompt": "Write a short haiku about GPUs and scheduling.",
    "max_new_tokens": 64,
    "temperature": 0.8,
    "top_p": 0.95,
}


def one() -> tuple[float, float]:
    """Return (ttft_ms, total_ms)."""
    t0 = time.perf_counter()
    ttft = None
    r = requests.post(URL, json=PAYLOAD, timeout=120, stream=True)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data: "):
            continue
        chunk = json.loads(decoded[len("data: "):])
        if chunk.get("is_first") and ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        if chunk.get("is_done"):
            total = (time.perf_counter() - t0) * 1000
            break
    else:
        total = (time.perf_counter() - t0) * 1000
    if ttft is None:
        ttft = total
    return ttft, total


def main(concurrency: int = 4, n: int = 20) -> None:
    ttfts = []
    totals = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one) for _ in range(n)]
        for f in as_completed(futs):
            ttft, total = f.result()
            ttfts.append(ttft)
            totals.append(total)

    ttfts.sort()
    totals.sort()

    def pct(vals: list[float], p: float) -> float:
        return vals[int(p * (len(vals) - 1))]

    summary = {
        "n": n,
        "concurrency": concurrency,
        "ttft_p50_ms": pct(ttfts, 0.50),
        "ttft_p90_ms": pct(ttfts, 0.90),
        "ttft_p99_ms": pct(ttfts, 0.99),
        "total_p50_ms": pct(totals, 0.50),
        "total_p90_ms": pct(totals, 0.90),
        "total_p99_ms": pct(totals, 0.99),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
