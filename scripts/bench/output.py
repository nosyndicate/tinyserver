from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = Path(args.out)
    suffix_parts = [
        f"scenario={args.scenario}",
        f"endpoint={args.endpoint}",
        f"mode={args.mode}",
    ]
    if args.mode == "closed":
        suffix_parts.append(f"concurrency={args.concurrency}")
    else:
        suffix_parts.append(f"arrival_rate={args.arrival_rate}")
    return base / ts / "/".join(suffix_parts)
