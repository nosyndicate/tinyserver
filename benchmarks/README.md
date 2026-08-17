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

Requests are offered at a fixed `--arrival-rate` (requests per second) regardless of how many are already in-flight. This exposes queue growth, backpressure, and tail latency under overload.

```bash
python -m scripts.bench \
  --scenario burst \
  --mode open \
  --arrival-rate 12 \
  --client-max-in-flight 256 \
  --duration-seconds 60
```

**Note:** `--arrival-rate` is required for open-loop mode. Exactly one of `--requests` or `--duration-seconds` must always be specified.

#### Arrival rate vs. client capacity

The two are separate knobs. `--arrival-rate` is the schedule the client *offers*;
`--client-max-in-flight` (default 256) bounds how many requests may be
outstanding at once. `--concurrency` plays no part in open loop — it is
closed-loop only.

Arrivals are fixed-interval, recorded as `arrival_process: "deterministic"` in
`config.json`, so a rerun offers requests at the same instants.

When the in-flight bound is reached the dispatcher blocks, and the wait shows up
as `client_dispatch_lag_ms` on every request offered afterwards. That is the
point: a client that cannot keep up says so instead of silently queueing
requests internally and reporting a load it never actually generated.

#### Client saturation

Every open-loop run reports its offer window in `summary.json`:

| Field | Meaning |
|---|---|
| `target_arrival_rate` | The rate requested via `--arrival-rate` |
| `achieved_arrival_rate` | Requests actually dispatched ÷ offer window |
| `offer_window_seconds` | First arrival → one interval past the last one, or the wall time taken if that was longer |
| `requested_duration_seconds` | What `--duration-seconds` asked for; `null` in `--requests` mode |
| `client_max_in_flight` | The capacity bound in force |
| `client_dispatch_lag_ms` | Percentiles (`mean`, `p50`, `p90`, `p95`, `p99`) over scheduled arrival → HTTP start |
| `completions_in_offer_window` | Requests that also *finished* before the window closed |
| `drain_seconds` | How long the tail of in-flight requests took after it closed |
| `client_saturated` | See below |

`client_saturated` is true when `achieved_arrival_rate` falls below 95% of
target, **or** when p95 dispatch lag exceeds one full arrival interval.

`offer_window_seconds` is **schedule-relative**: each request owns one arrival
interval, so N requests occupy N intervals. This is what makes
`achieved_arrival_rate` come out at exactly the target for a client that keeps
up, and drop below it for one that does not. A consequence is that in duration
mode the window can exceed `--duration-seconds` by up to one interval, whenever
`duration × arrival-rate` is not a whole number — compare it against
`requested_duration_seconds`. The window is deliberately not capped at the
deadline: that would shorten the denominator without changing the request count,
making the achieved rate read *above* target. `--duration-seconds` shorter than
one arrival interval is rejected outright, since it cannot sample the rate at
all.

**A saturated cell measures the benchmark client, not the server.** Its latency
and throughput numbers are not comparable with an unsaturated cell and must not
be ranked against one. Raise `--client-max-in-flight`, or lower the arrival rate,
and rerun.

---

## CLI reference

### Server

| Flag | Default | Description |
|---|---|---|
| `--base-url` | `http://127.0.0.1:8000` | Server address |
| `--endpoint` | `stream_v2` | API endpoint: `generate`, `generate_v2`, `stream`, `stream_v2`, `generate/stream`, `generate/stream_v2` |

**Note:** each server mode only serves its own endpoints. The v1 endpoints
(`generate`, `stream`) require a server started with `--api-version v1`; the
versioned endpoints (`*_v2`/`*_v3`/`*_v4`) require the matching mode. To
compare v1 against v2 you must run the server twice, once per mode.
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
| `--concurrency` | `4` | Worker thread count — **closed loop only** |
| `--arrival-rate` | — | Requests offered per second (required for open-loop) |
| `--client-max-in-flight` | `256` | Open loop: maximum outstanding requests. Bounds client capacity independently of the arrival rate |
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
| `config.json` | Exact CLI arguments, the run's clock epochs, and the resolved scenario definition; sufficient to reproduce the run |
| `requests.jsonl` | One JSON object per request (including failures); raw data for post-hoc analysis |

### Corrected measurement semantics (PR 94)

The definitions below replace measurements that were producing invalid numbers —
wall-clock timestamps, a client token count that dropped empty decodes, a TPOT
that divided a server-side numerator by a client-side count, and a `latency_ms`
overwritten with the server's `total_ms`. They are a correction, not a new
generation of the format.

Artifacts written before the correction are told apart by what they lack: no
`client_*` or `server_*` fields, no `output_tokens_source`, no `output_sha256`.

**All timing is monotonic.** A single `perf_counter` epoch is taken at run start;
every `*_ts` and `*_offset_s` field is seconds relative to that epoch, *not* a
Unix timestamp. `config.json` records both halves of the epoch
(`clock.wall_epoch_s`, `clock.wall_epoch_iso`, `clock.perf_epoch_s`) so offsets
can be mapped back to a real date. Wall clock is never used for a duration, so
NTP slew cannot corrupt a latency measurement.

**Client and server metrics are kept apart.** The two are measured on different
clocks from different origins, so combining them in one number is meaningless.

| Field | Meaning |
|---|---|
| `latency_ms` | Client: scheduled arrival → completion. The headline end-to-end number |
| `client_dispatch_lag_ms` | Client: scheduled arrival → HTTP start. `0.0` in closed loop, where arrival *is* the HTTP start; in open loop it is the client's own queueing delay, and it drives `client_saturated` |
| `scheduled_arrival_offset_s` | Client: the instant the open-loop schedule called for this request, from the run epoch |
| `client_http_ms` | Client: HTTP start → completion |
| `client_ttft_ms` | Client: HTTP start → first streamed token |
| `client_tpot_ms` | `(client_http_ms − client_ttft_ms) / (n − 1)`; `null` below two tokens |
| `server_total_ms` | Server: enqueue → completion |
| `server_queue_wait_ms` | Server: enqueue → first prefill |
| `server_execution_ms` | Server: first prefill → completion |
| `server_ttft_ms` | Server: enqueue → first token. `null` when no token was emitted |
| `server_tpot_ms` | `(server_execution_ms − (server_ttft_ms − server_queue_wait_ms)) / (n − 1)`; `null` below two tokens |
| `output_tokens` | The server's authoritative count from the terminal `done` event |
| `output_tokens_source` | `server` for successful streams; `client_count` only for partial failed streams |
| `output_sha256` | SHA-256 over the concatenated streamed tokens |
| `deterministic_gate` | `true` when `temperature == 0.0` and a `seed` was set |

`ttft_ms`, `tpot_ms`, `queue_wait_ms`, and `execution_ms` are retained as aliases
of `client_ttft_ms`, `client_tpot_ms`, `server_queue_wait_ms`, and
`server_execution_ms` so existing analysis scripts keep loading.

**Non-streaming endpoints report `client_ttft_ms` and `client_tpot_ms` as
`null`.** The whole response arrives at once, so there is no client-observable
first token; use `server_ttft_ms` / `server_tpot_ms` for those runs. The server's
figure is not copied into the client field, because it is measured from server
enqueue on a different clock.

**Output tokens come from the server's explicit stream protocol.** Each sampled
non-EOS token produces a `type: "token"` event, even when its decoded
`token_str` is `""`; EOS produces no token event. A separate `type: "done"`
event reports the authoritative count, finish reason, and timing metrics. The
client verifies contiguous zero-based token indices and requires the done count
to match the number of observed token events, eliminating terminal-chunk
heuristics across v1/v2/v3/v4.

**`output_sha256` is only comparable where `deterministic_gate` is true.** At
`temperature > 0`, v2/v3/v4 will not produce byte-identical text across batch
shapes, so on ungated rows the digest is informational and only token counts are
worth sanity-checking.

> **Compatibility:** `latency_ms` was previously overwritten with the server's
> `total_ms`, and `start_ts` / `first_token_ts` / `end_ts` were wall-clock Unix
> timestamps. Artifacts produced before this correction are **not** comparable
> with those produced after it, and must not be merged into one table.

### Server-reported timing semantics

The server reports four timings per request, each measured between two raw
timestamps so that `total_ms == queue_wait_ms + execution_ms`:

```
enqueue ---- queue_wait_ms ----> first prefill ---- execution_ms ----> completion
|<---------------------------- total_ms --------------------------------->|
```

- `ttft_ms` is measured **from enqueue**, so it includes queue wait. It is
  `null` when a request completes without emitting a token.
- `tokens_per_s` is the decode rate: output tokens over `execution_ms`. It
  deliberately excludes queue wait, so it stays comparable across load levels.
  System-level throughput is a separate figure, computed by the benchmark
  client over its measurement window and reported in `summary.json`.
- When preemption available, `execution_ms` includes any time a sequence spent 
  preempted. The start of the *first* prefill is not reset on resume, so preemption 
  cost is accounted as execution rather than as queue wait.

> **Compatibility:** these definitions changed. Previously `total_ms` and
> `ttft_ms` were measured from the start of prefill rather than from enqueue,
> `execution_ms` was computed as `total_ms - queue_wait_ms` (which
> under-reported it by exactly the queue wait, clamping to `0.0` under load),
> and `ttft_ms` used a `-1.0` sentinel instead of `null`. Artifacts produced
> before this change are **not** comparable with artifacts produced after it.

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
    --client-max-in-flight 256 \
    --duration-seconds 30 \
    --out bench-results
done
```

Check `client_saturated` in each `summary.json` before comparing the cells; at
the top of the sweep the client can run out of capacity before the server does,
and such a cell says nothing about the server.

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
