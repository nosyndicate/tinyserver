"""App Entrypoint"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI

from server.api.routes import router as api_router
from server.executor.executor import BatchExecutor, Executor
from server.executor.types import BatchExecutorConfig, ExecutorConfig
from server.executor.worker import BatchWorker, SimpleWorker
from server.model.hf_backend import HFBackend
from server.model.hf_runner import ModelConfig, ModelRunner


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the lifecycle of the model, executor and worker.
    """

    # Everything before the yield runs on startup
    config = ModelConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    backend = HFBackend.load_model(config)

    runner = ModelRunner(
        model=backend.model, tokenizer=backend.tokenizer, device=config.device
    )

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


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Inference Server (Phase 2)", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
