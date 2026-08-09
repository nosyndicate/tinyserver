"""Regression coverage for async handlers under threadpool-saturating load."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import asynccontextmanager
from queue import Empty

import pytest
import requests
import uvicorn
from fastapi import FastAPI

from server.api.collector import CollectorRegistry
from server.api.pump import OutputPump
from server.api.routes import health_router, v4_router
from server.executor.events import RequestEventEmitter
from server.executor.sinks import SharedQueueSink
from server.executor.types import DecodeResult, FinishReason, GenerationRequestState
from server.executor.worker import Worker
from server.metrics.timers import now_ns

pytestmark = pytest.mark.timing


# The test only means something if it oversubscribes anyio's threadpool: with
# sync handlers, requests beyond the worker count wait for a free thread, so
# _REQUESTS holds take ceil(_REQUESTS / 40) * _HOLD_S seconds instead of one
# _HOLD_S window. 80 > 40 by 2x, so the pre-fix shape needs >= 2 windows (4.0s)
# while the async path finishes in one (~2.0s) - a 1s gap either side of the
# 3.0s limit.
_REQUESTS = 80
_HOLD_S = 2.0
_TOTAL_WALL_TIME_LIMIT_S = _HOLD_S * 1.5
_HEALTH_TIMEOUT_S = 1.0


class _SlowFakeEngine:
    """CPU-only engine that keeps many requests pending concurrently."""

    def __init__(self) -> None:
        self._emitter = RequestEventEmitter()
        self._active: dict[str, GenerationRequestState] = {}
        self._lock = threading.Lock()

    def run(self, inbound, control, callbacks) -> None:  # noqa: ANN001
        while not control.should_stop():
            try:
                request = inbound.get(timeout=0.05)
            except Empty:
                if control.wait_idle(0.05):
                    break
                continue
            self._emitter.on_prefill_started(request, now_ns())
            request.num_prompt_tokens = 1
            with self._lock:
                self._active[request.request_id] = request
            threading.Thread(
                target=self._finish_after_delay, args=(request,), daemon=True
            ).start()

    def _finish_after_delay(self, request: GenerationRequestState) -> None:
        time.sleep(_HOLD_S)
        with self._lock:
            self._active.pop(request.request_id, None)
        if request.cancelled.is_set():
            return
        self._emitter.on_token(
            request,
            DecodeResult(token_id=1, token="x", finish_reason=FinishReason.MAX_LENGTH),
        )

    def cancel_inflight(self, message, cancel_request) -> None:  # noqa: ANN001
        with self._lock:
            for request in self._active.values():
                request.cancelled.set()
                cancel_request(request, message)
            self._active.clear()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    registry = CollectorRegistry()
    sink = SharedQueueSink()
    worker = Worker(_SlowFakeEngine(), max_queue_size=256)
    worker.start()
    app.state.worker = worker
    app.state.registry = registry
    app.state.sink = sink
    app.state.device = "cpu"
    app.state.runner = None
    pump = OutputPump(sink.queue, registry)
    pump.start(asyncio.get_running_loop())
    try:
        yield
    finally:
        worker.stop()
        await pump.stop_and_flush()
        registry.fail_all("Server is shutting down")


def _build_app() -> FastAPI:
    app = FastAPI(lifespan=_lifespan)
    app.include_router(health_router)
    app.include_router(v4_router)
    return app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server() -> tuple[uvicorn.Server, threading.Thread, str]:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(_build_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=0.25).status_code == 200:
                return server, thread, base_url
        except requests.RequestException:
            pass
        time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    raise RuntimeError("test server did not become healthy within 15 seconds")


def test_health_remains_responsive_during_concurrent_async_generations() -> None:
    server, server_thread, base_url = _start_server()
    results: list[int | None] = [None] * _REQUESTS
    payload = {"prompt": "hi", "max_new_tokens": 1, "temperature": 1.0, "top_p": 0.95}

    def generate(index: int) -> None:
        try:
            results[index] = requests.post(
                f"{base_url}/generate_v4", json=payload, timeout=15
            ).status_code
        except requests.RequestException:
            results[index] = None

    clients = [
        threading.Thread(target=generate, args=(index,)) for index in range(_REQUESTS)
    ]
    try:
        started = time.monotonic()
        for client in clients:
            client.start()
        time.sleep(0.5)
        probe_started = time.monotonic()
        health = requests.get(f"{base_url}/health", timeout=_HEALTH_TIMEOUT_S)
        health_elapsed = time.monotonic() - probe_started
        for client in clients:
            client.join(timeout=15)
        wall_time = time.monotonic() - started

        assert health.status_code == 200
        assert health_elapsed < _HEALTH_TIMEOUT_S
        assert results == [200] * _REQUESTS
        assert wall_time < _TOTAL_WALL_TIME_LIMIT_S
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        assert not server_thread.is_alive(), "test server did not stop"
