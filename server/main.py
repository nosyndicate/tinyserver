"""App Entrypoint"""

import argparse
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI

from server.api.routes import (
    health_router,
    v2_router,
    v3_router,
    v4_router,
)
from server.api.v1 import v1_router
from server.executor.engine import (
    BatchInferenceEngine,
    ScheduleInferenceEngine,
    SimpleInferenceEngine,
)
from server.executor.executor import BatchExecutor, Executor
from server.executor.scheduler import Scheduler
from server.executor.types import BatchEngineConfig, EngineConfig
from server.executor.worker import Worker
from server.model.block_manager import BlockManager
from server.model.hf_backend import HFBackend
from server.model.hf_runner import ModelConfig, load_hf_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference server."""
    parser = argparse.ArgumentParser(description="LLM Inference Server")
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Size of each KV cache block in tokens (default: 256)",
    )
    parser.add_argument(
        "--memory-utilization",
        type=float,
        default=0.2,
        help="Fraction of free GPU memory to use for KV cache (default: 0.2)",
    )
    parser.add_argument(
        "--api-version",
        choices=["v1", "v2", "v3", "v4"],
        default="v3",
        help=(
            "Which endpoint version to expose (default: v3). Each mode serves "
            "only its own endpoints (plus /health). v1 is the preserved "
            "phase-1 baseline (direct, unqueued) and is only available in v1 "
            "mode."
        ),
    )
    parser.add_argument(
        "--max-waiting",
        type=int,
        default=64,
        help="v4 only: max sequences in the scheduler's waiting queue (default: 64)",
    )
    parser.add_argument(
        "--max-num-sequences",
        type=int,
        default=8,
        help="v4 only: max sequences per scheduled batch (default: 8)",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=4096,
        help="v4 only: max total tokens per scheduled batch (default: 4096)",
    )
    parser.add_argument(
        "--v3-max-decode-batch-size",
        type=int,
        default=8,
        help="v3 only: max sequences per decode batch (default: 8)",
    )
    parser.add_argument(
        "--v3-max-prefill-batch-size",
        type=int,
        default=8,
        help="v3 only: max sequences per prefill batch (default: 8)",
    )
    parser.add_argument(
        "--v3-max-active-requests",
        type=int,
        default=16,
        help="v3 only: max concurrent requests in the engine (default: 16)",
    )
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the lifecycle of the model, executor and worker.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is not available. This server requires a GPU to run."
        )

    # Everything before the yield runs on startup
    args = app.state.cli_args
    config = ModelConfig(
        device="cuda",
        block_size=args.block_size,
        memory_utilization=args.memory_utilization,
    )

    version = args.api_version

    app.state.runner = None
    app.state.device = config.device
    app.state.worker = None

    if version == "v4":
        # v4 uses the paged-attention HFBackend instead of ModelRunner; the
        # runner is skipped entirely so only one copy of the model is loaded.
        backend = HFBackend.load_model(config)
        block_manager = BlockManager(
            total_blocks=backend.num_blocks,
            block_size=backend.block_size,
        )
        scheduler = Scheduler(
            block_manager,
            max_waiting=args.max_waiting,
            max_num_sequences=args.max_num_sequences,
            max_num_tokens=args.max_num_tokens,
        )
        worker = Worker(
            ScheduleInferenceEngine(scheduler, backend),
            max_queue_size=64,
        )
        worker.start()
        app.state.worker = worker

        yield

        worker.stop()
        return

    runner = load_hf_model(config)
    if version == "v1":
        app.state.runner = runner

    if version == "v2":
        executor = Executor(runner)
        engine_config = EngineConfig()
        worker = Worker(
            SimpleInferenceEngine(executor, engine_config),
            max_queue_size=64,
        )
        worker.start()
        app.state.worker = worker
    elif version == "v3":
        batch_executor = BatchExecutor(runner)
        engine_config = BatchEngineConfig(
            max_active_requests=args.v3_max_active_requests,
            max_prefill_batch_size=args.v3_max_prefill_batch_size,
            max_decode_batch_size=args.v3_max_decode_batch_size,
        )
        worker = Worker(
            BatchInferenceEngine(batch_executor, engine_config),
            max_queue_size=64,
        )
        worker.start()
        app.state.worker = worker

    # App is running now
    yield

    # Everything after the yield runs on shutdown.
    if app.state.worker is not None:
        app.state.worker.stop()


def create_app(cli_args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="LLM Inference Server", lifespan=lifespan)
    app.state.cli_args = cli_args
    app.include_router(health_router)
    if cli_args.api_version == "v1":
        app.include_router(v1_router)
    elif cli_args.api_version == "v2":
        app.include_router(v2_router)
    elif cli_args.api_version == "v3":
        app.include_router(v3_router)
    elif cli_args.api_version == "v4":
        app.include_router(v4_router)
    return app


def main() -> None:
    """Main entry point for the inference server."""
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
