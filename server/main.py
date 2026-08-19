"""App Entrypoint"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI

from server.api.collector import CollectorRegistry
from server.api.pump import OutputPump
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
    validate_batch_engine_config,
)
from server.executor.executor import BatchExecutor, Executor
from server.executor.scheduler import Scheduler
from server.executor.sinks import SharedQueueSink
from server.executor.types import BatchEngineConfig, EngineConfig
from server.executor.worker import Worker
from server.model.block_manager import BlockManager
from server.model.hf_backend import HFBackend
from server.model.hf_runner import ModelConfig, load_hf_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the inference server.

    Each API version is its own subcommand, so a version can only be given the
    flags it actually reads.
    """
    # Shared flags live on parent parsers so they may follow the subcommand
    # (``server.main v4 --host …``) rather than having to precede it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address to bind the server to (default: 0.0.0.0)",
    )
    common.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )

    # v1 is direct and unqueued, so it has no worker to size.
    worker = argparse.ArgumentParser(add_help=False)
    worker.add_argument(
        "--worker-queue-size",
        type=int,
        default=64,
        help="Max requests buffered in the worker's inbound queue (default: 64)",
    )

    parser = argparse.ArgumentParser(description="LLM Inference Server")
    versions = parser.add_subparsers(
        dest="api_version",
        required=True,
        metavar="{v1,v2,v3,v4}",
        help="Which endpoint version to expose. Each serves only its own "
        "endpoints, plus /health.",
    )

    versions.add_parser(
        "v1",
        parents=[common],
        help="Preserved phase-1 baseline: direct and unqueued, no worker.",
    )

    v2 = versions.add_parser(
        "v2",
        parents=[common, worker],
        help="Queue-backed worker, one request at a time per slot.",
    )
    v2.add_argument(
        "--max-active-requests",
        type=int,
        default=16,
        help="Max requests the engine processes concurrently (default: 16)",
    )

    v3 = versions.add_parser(
        "v3",
        parents=[common, worker],
        help="Batched prefill and decode.",
    )
    v3.add_argument(
        "--max-active-requests",
        type=int,
        default=16,
        help="Max requests the engine processes concurrently (default: 16)",
    )
    v3.add_argument(
        "--max-prefill-batch-size",
        type=int,
        default=8,
        help="Max requests per prefill batch (default: 8)",
    )
    v3.add_argument(
        "--max-decode-batch-size",
        type=int,
        default=8,
        help="Max requests per decode batch (default: 8)",
    )

    v4 = versions.add_parser(
        "v4",
        parents=[common, worker],
        help="Continuous batching over a paged KV cache.",
    )
    # The KV-pool knobs are v4's alone: it is the only version with a paged
    # pool (see model/hf_backend.py).
    v4.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Size of each KV cache block in tokens (default: 256)",
    )
    v4.add_argument(
        "--memory-utilization",
        type=float,
        default=0.2,
        help="Fraction of free GPU memory to use for KV cache (default: 0.2)",
    )
    v4.add_argument(
        "--max-waiting",
        type=int,
        default=64,
        help="Max sequences in the scheduler's waiting queue (default: 64)",
    )
    v4.add_argument(
        "--max-num-sequences",
        type=int,
        default=8,
        help="Max sequences per scheduled batch (default: 8)",
    )
    v4.add_argument(
        "--max-num-tokens",
        type=int,
        default=4096,
        help="Max total tokens per scheduled batch (default: 4096)",
    )

    args = parser.parse_args(argv)
    # Checked here rather than at worker construction so an impossible batch
    # combination fails before the model is loaded, not several seconds into
    # startup. Each branch validates exactly the flags its version accepts.
    try:
        if args.api_version == "v3":
            validate_batch_engine_config(_batch_engine_config(args))
        elif args.api_version == "v2" and args.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _model_config(args: argparse.Namespace) -> ModelConfig:
    """The model configuration implied by the parsed arguments.

    The KV-pool knobs exist only under v4, the only version with a paged pool;
    the others take the dataclass defaults, which are the same values their
    flags used to default to.
    """
    if args.api_version == "v4":
        return ModelConfig(
            device="cuda",
            block_size=args.block_size,
            memory_utilization=args.memory_utilization,
        )
    return ModelConfig(device="cuda")


def _engine_config(args: argparse.Namespace) -> EngineConfig:
    """The v2 engine configuration implied by the parsed arguments."""
    return EngineConfig(max_active_requests=args.max_active_requests)


def _batch_engine_config(args: argparse.Namespace) -> BatchEngineConfig:
    """The v3 engine configuration implied by the parsed arguments."""
    return BatchEngineConfig(
        max_active_requests=args.max_active_requests,
        max_prefill_batch_size=args.max_prefill_batch_size,
        max_decode_batch_size=args.max_decode_batch_size,
    )


def _build_worker(
    version: str, config: ModelConfig, args: argparse.Namespace
) -> Worker:
    """Construct the queue-backed worker for one API version."""
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
        return Worker(
            ScheduleInferenceEngine(scheduler, backend),
            max_queue_size=args.worker_queue_size,
        )

    runner = load_hf_model(config)
    if version == "v2":
        return Worker(
            SimpleInferenceEngine(Executor(runner), _engine_config(args)),
            max_queue_size=args.worker_queue_size,
        )
    if version == "v3":
        return Worker(
            BatchInferenceEngine(BatchExecutor(runner), _batch_engine_config(args)),
            max_queue_size=args.worker_queue_size,
        )
    raise ValueError(f"No queue-backed worker for api version {version!r}")


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
    config = _model_config(args)

    version = args.api_version

    app.state.runner = None
    app.state.device = config.device
    app.state.worker = None
    app.state.registry = None
    app.state.sink = None

    pump: OutputPump | None = None

    if version == "v1":
        # The preserved phase-1 baseline: direct, unqueued, no worker to pump.
        app.state.runner = load_hf_model(config)
    else:
        registry = CollectorRegistry()
        sink = SharedQueueSink()
        app.state.registry = registry
        app.state.sink = sink

        worker = _build_worker(version, config, args)
        worker.start()
        app.state.worker = worker

        # lifespan itself runs on the default event loop, so this is where the loop
        # reference is captured — once, rather than per request.
        pump = OutputPump(sink.queue, registry)
        pump.start(asyncio.get_running_loop())

    # App is running now
    yield

    # Everything after the yield runs on shutdown. The order matters:
    # `Worker.stop()` drains its inbound queue and cancels in-flight requests,
    # emitting a final ErrorEvent per request through the shared sink — so the
    # pump must still be alive to route them. `pump.stop_and_flush()` joins the
    # thread, drains remaining events, and runs their dispatch callbacks before
    # `fail_all` wakes anyone still waiting.
    if app.state.worker is not None:
        app.state.worker.stop()
    if pump is not None:
        await pump.stop_and_flush()
    if app.state.registry is not None:
        app.state.registry.fail_all("Server is shutting down")


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
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
