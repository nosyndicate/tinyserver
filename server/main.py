"""App Entrypoint"""

import argparse
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI

from server.api.routes import router as api_router
from server.executor.executor import BatchExecutor, Executor
from server.executor.types import BatchExecutorConfig, ExecutorConfig
from server.executor.worker import BatchWorker, SimpleWorker
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

    # runner and backend will load two models in total.
    # but the model load in backend will be modified to support kv cache
    # model load by load_hf_model will be used to support v2 and v3
    runner = load_hf_model(config)
    HFBackend.load_model(config)

    executor = Executor(runner)
    executor_config = ExecutorConfig()
    worker = SimpleWorker(executor, executor_config)
    worker.start()

    batch_executor = BatchExecutor(runner)
    batch_executor_config = BatchExecutorConfig()
    batch_worker = BatchWorker(batch_executor, batch_executor_config)
    batch_worker.start()

    app.state.runner = runner
    app.state.worker = worker
    app.state.batch_worker = batch_worker
    app.state.device = config.device

    # App is running now
    yield

    # Everything after the yield runs on shutdown.
    worker.stop()
    batch_worker.stop()


def create_app(cli_args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="LLM Inference Server (Phase 2)", lifespan=lifespan)
    app.state.cli_args = cli_args
    app.include_router(api_router)
    return app


def main() -> None:
    """Main entry point for the inference server."""
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
