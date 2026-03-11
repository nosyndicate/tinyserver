import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

URL = "http://127.0.0.1:8000/generate"

PAYLOAD = {
    "prompt": "Write a short haiku about GPUs and scheduling.",
    "max_new_tokens": 64,
    "temperature": 0.8,
    "top_p": 0.95,
}


def one() -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = requests.post(URL, json=PAYLOAD, timeout=120)
    r.raise_for_status()
    dt = (time.perf_counter() - t0) * 1000
    return dt, r.json()


def main(concurrency: int = 4, n: int = 20) -> None:
    lat = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one) for _ in range(n)]
        for f in as_completed(futs):
            dt, out = f.result()
            lat.append(dt)

    lat.sort()

    def pct(p: float) -> float:
        return lat[int(p * (len(lat) - 1))]

    summary = {
        "n": n,
        "concurrency": concurrency,
        "ttft_p50_ms": pct(0.50),
        "ttft_p90_ms": pct(0.90),
        "ttft_p99_ms": pct(0.99),
        "total_p50_ms": pct(0.50),
        "total_p90_ms": pct(0.90),
        "total_p99_ms": pct(0.99),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
