from __future__ import annotations

import sys

from bench import main as bench_main


if __name__ == "__main__":
    argv = ["--endpoint", "stream_v2"] + sys.argv[1:]
    raise SystemExit(bench_main(argv))
