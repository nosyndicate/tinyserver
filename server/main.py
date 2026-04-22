"""App Entrypoint"""

import argparse
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI

from server.api.routes import health_router, v1_router, v2_router, v3_router
from server.executor.engine import BatchInferenceEngine, SimpleInferenceEngine
from server.executor.executor import BatchExecutor, Executor
from server.executor.types import BatchExecutorConfig, ExecutorConfig
from server.executor.worker import Worker
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
        choices=["v1", "v2", "v3"],
        default="v3",
        help="Which endpoint version to expose (default: v3). v1 endpoints are always included.",
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

    runner = load_hf_model(config)
    app.state.runner = runner
    app.state.device = config.device
    app.state.worker = None
    app.state.batch_worker = None

    version = args.api_version

    if version == "v2":
        executor = Executor(runner)
        config = ExecutorConfig()
        worker = Worker(SimpleInferenceEngine(executor, config), config)
        worker.start()
        app.state.worker = worker
    elif version == "v3":
        batch_executor = BatchExecutor(runner)
        config = BatchExecutorConfig()
        batch_worker = Worker(BatchInferenceEngine(batch_executor, config), config)
        batch_worker.start()
        app.state.batch_worker = batch_worker

    # App is running now
    yield

    # Everything after the yield runs on shutdown.
    if app.state.worker is not None:
        app.state.worker.stop()
    if app.state.batch_worker is not None:
        app.state.batch_worker.stop()


def create_app(cli_args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="LLM Inference Server", lifespan=lifespan)
    app.state.cli_args = cli_args
    app.include_router(health_router)
    app.include_router(v1_router)
    if cli_args.api_version == "v2":
        app.include_router(v2_router)
    elif cli_args.api_version == "v3":
        app.include_router(v3_router)
    return app


def main() -> None:
    """Main entry point for the inference server."""
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
