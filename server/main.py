"""App Entrypoint"""

from fastapi import FastAPI

from server.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Inference Server (Phase 0)")
    app.include_router(api_router)
    return app


app = create_app()
