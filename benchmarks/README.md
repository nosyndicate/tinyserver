# Benchmarks

The benchmarking framework is a CLI tool at `scripts/bench/` that generates synthetic load against the inference server and collects structured per-request performance data. It replaces the original `bench.py` / `bench_stream.py` scripts with a single package that supports scenario-driven testing, closed-loop and open-loop load generation, and durable machine-readable output artifacts.

All commands below are run from the **repository root**.

---

## Prerequisites

- The inference server must be running (default `http://127.0.0.1:8000`).
- The `requests` package must be installed in your Python environment.

---

## Quick start

Single-request smoke test, print summary to stdout:

```bash
python -m scripts.bench \
  --endpoint stream_v2 \
  --scenario short_short \
  --requests 1 \
  --summary-only
```

Closed-loop concurrency run, write results to disk:

```bash
python -m scripts.bench \
  --endpoint stream_v2 \
  --scenario mixed \
  --mode closed \
  --concurrency 16 \
  --requests 200 \
  --warmup-requests 10 \
  --out bench-results
```

---

## Scenarios

The default scenario file is `benchmarks/scenarios.json`. Five named scenarios are included:

| Scenario | Prompt file | `max_new_tokens` | Purpose |
|---|---|---|---|
| `short_short` | `prompts/short.txt` | 48 | TTFT-focused; minimal prefill and fast decode |
| `long_long` | `prompts/long.txt` | 192 | Prefill and throughput stress |
| `mixed` | short (weight 3) + long (weight 1) | 48 / 192 | Realistic traffic mix |
| `burst` | `prompts/burst.txt` | 96 | Open-loop queue pressure |
| `seeded_deterministic` | `prompts/deterministic.txt` | 64 (temp=0, seed=7) | Regression and determinism checks |

The `mixed` scenario cycles through a 3:1 short-to-long request pattern regardless of total request count.

---

## Load modes

### Closed-loop (`--mode closed`)

A fixed pool of `--concurrency` worker threads runs continuously: each worker submits the next request only after its previous one completes. This measures the maximum sustainable throughput at a given in-flight count.

```bash
python -m scripts.bench \
  --scenario short_short \
  --mode closed \
  --concurrency 8 \
  --requests 100
```

### Open-loop (`--mode open`)

Requests are dispatched at a fixed `--arrival-rate` (requests per second) regardless of how many are already in-flight. This exposes queue growth, backpressure, and tail latency under overload.

```bash
python -m scripts.bench \
  --scenario burst \
  --mode open \
  --arrival-rate 12 \
  --duration-seconds 60
```

**Note:** `--arrival-rate` is required for open-loop mode. Exactly one of `--requests` or `--duration-seconds` must always be specified.

---

## CLI reference

### Server

| Flag | Default | Description |
|---|---|---|
| `--base-url` | `http://127.0.0.1:8000` | Server address |
| `--endpoint` | `stream_v2` | API endpoint: `generate`, `generate_v2`, `stream`, `stream_v2`, `generate/stream`, `generate/stream_v2` |
| `--timeout-seconds` | `120.0` | Per-request HTTP timeout |

### Scenario

| Flag | Default | Description |
|---|---|---|
| `--scenario` | `short_short` | Named scenario to run |
| `--scenario-file` | `benchmarks/scenarios.json` | Path to a custom scenario JSON (merges with built-ins) |
| `--prompt-file` | — | Override all prompts with text from a file |
| `--max-new-tokens` | — | Override scenario output token limit |
| `--temperature` | — | Override sampling temperature |
| `--top-p` | — | Override nucleus sampling threshold |
| `--seed` | — | Override random seed |

CLI overrides take precedence over scenario values; unset flags leave scenario values intact.

### Load control

| Flag | Default | Description |
|---|---|---|
| `--mode` | `closed` | `closed` (fixed concurrency) or `open` (fixed arrival rate) |
| `--concurrency` | `4` | Worker thread count (both modes) |
| `--arrival-rate` | — | Requests per second (required for open-loop) |
| `--requests` | — | Total measurement requests (mutually exclusive with `--duration-seconds`) |
| `--duration-seconds` | — | Measurement window in seconds (mutually exclusive with `--requests`) |
| `--warmup-requests` | `0` | Sequential warmup requests excluded from results |

### Output

| Flag | Default | Description |
|---|---|---|
| `--out` | `bench-results` | Root directory for output artifacts |
| `--summary-only` | off | Print summary JSON to stdout; skip writing files |

---

## Output artifacts

Each run writes to a path that encodes its configuration:

```
bench-results/
  2025-04-02T143000Z/
    scenario=short_short/
      endpoint=stream_v2/
        mode=closed/
          concurrency=4/
            summary.json
            config.json
            requests.jsonl
```

| File | Contents |
|---|---|
| `summary.json` | Aggregate statistics: request counts, success rate, throughput (req/s, tokens/s), and P50/P90/P95/P99 distributions for `latency_ms`, `ttft_ms`, `tpot_ms`, `queue_wait_ms`, `execution_ms` |
| `config.json` | Exact CLI arguments and the resolved scenario definition; sufficient to reproduce the run |
| `requests.jsonl` | One JSON object per request (including failures); raw data for post-hoc analysis |

---

## Common recipes

### Phase 1 — smoke test

Verify the server responds correctly on both endpoints:

```bash
# Streaming
python -m scripts.bench --endpoint stream_v2 --scenario short_short --requests 1 --summary-only

# Sync
python -m scripts.bench --endpoint generate_v2 --scenario short_short --requests 1 --summary-only
```

### Phase 2 — concurrency sweep

Run the same scenario at increasing concurrency levels to find saturation:

```bash
for C in 1 4 8 16 32; do
  python -m scripts.bench \
    --endpoint stream_v2 \
    --scenario short_short \
    --mode closed \
    --concurrency $C \
    --requests 100 \
    --warmup-requests 5 \
    --out bench-results
done
```

### Phase 2 — saturation sweep (open-loop)

Step up arrival rate until 503s appear or latency blows up:

```bash
for R in 2 4 8 16 32; do
  python -m scripts.bench \
    --endpoint stream_v2 \
    --scenario burst \
    --mode open \
    --arrival-rate $R \
    --duration-seconds 30 \
    --out bench-results
done
```

### Determinism check

Verify identical outputs for identical inputs across runs:

```bash
python -m scripts.bench \
  --scenario seeded_deterministic \
  --endpoint generate_v2 \
  --requests 5 \
  --warmup-requests 1 \
  --summary-only
```

### Custom scenario file

Override a scenario or add a new one without editing `benchmarks/scenarios.json`:

```bash
python -m scripts.bench \
  --scenario-file /tmp/my_scenarios.json \
  --scenario my_custom \
  --mode closed \
  --concurrency 4 \
  --requests 50
```

---

## Custom scenarios

Pass a JSON file via `--scenario-file`. Entries **merge** with the built-ins: matching names replace the default, new names are added.

```json
{
  "my_custom": {
    "description": "Custom scenario for prefix-cache experiments.",
    "requests": [
      {
        "prompt": "Summarize the tradeoffs of KV cache reuse.",
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "weight": 1,
        "metadata": {"class": "custom"}
      }
    ]
  }
}
```

Prompts can also reference files via `"prompt_file"`. The path is resolved relative to the scenario JSON's parent directory:

```json
{
  "long_context": {
    "requests": [{"prompt_file": "prompts/long.txt", "max_new_tokens": 256}]
  }
}
```

---

## Inspecting results

Print the top-level summary fields:

```bash
jq '{success_rate, request_throughput_rps, output_token_throughput_tps}' \
  bench-results/*/summary.json
```

Show TTFT and total-latency percentiles:

```bash
jq '{ttft_ms, latency_ms}' bench-results/*/summary.json
```

Filter failed requests from `requests.jsonl`:

```bash
jq 'select(.ok == false)' bench-results/*/requests.jsonl
```

Extract all output token counts from a run:

```bash
jq '.output_tokens' bench-results/*/requests.jsonl | sort -n
```
