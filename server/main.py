"""App Entrypoint"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from server.api.routes import router as api_router
from server.executor.executor import Executor
from server.executor.types import ExecutorConfig
from server.executor.worker import Worker
from server.model.hf_runner import ModelConfig, load_hf_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Manage the lifecycle of the model, executor and worker.
    """

    # Everything before the yield runs on startup
    config = ModelConfig()
    runner = load_hf_model(config)

    executor = Executor(runner)
    executor_config = ExecutorConfig()
    worker = Worker(executor, executor_config)
    worker.start()

    app.state.worker = worker
    app.state.device = config.device

    # App is running now
    yield

    # Everything after the yield runs on shutdown.
    worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Inference Server (Phase 2)")
    app.include_router(api_router)
    return app


app = create_app()
