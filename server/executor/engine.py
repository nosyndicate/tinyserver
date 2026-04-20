import logging
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable

from server.executor.events import RequestEventEmitter
from server.executor.types import (
    BaseBatchExecutor,
    BaseExecutor,
    BatchExecutorConfig,
    DecodeResult,
    ExecutorConfig,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
    RequestStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineControl:
    should_stop: Callable[[], bool]
    wait_idle: Callable[[float], bool]


@dataclass(frozen=True)
class EngineCallbacks:
    cancel_request: Callable[[GenerationRequestState, str], None]
    handle_fatal_error: Callable[[Exception, list[GenerationRequestState] | None], None]


class SimpleInferenceEngine:
    def __init__(self, executor: BaseExecutor, config: ExecutorConfig) -> None:
        self._executor = executor
        self._config = config
        self._active: list[GenerationRequestState] = []
        self._emitter = RequestEventEmitter()

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        for pending in self._active:
            try:
                cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for active request %s",
                    pending.request_id,
                )
        self._active.clear()

    def _cancel_requests(
        self,
        requests: list[GenerationRequestState],
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
        phase: str,
    ) -> None:
        for pending in requests:
            try:
                cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to clean up request %s during %s",
                    pending.request_id,
                    phase,
                )

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        try:
            while not control.should_stop():
                new_requests: list[GenerationRequestState] = []
                while (
                    len(self._active) + len(new_requests)
                    < self._config.max_active_requests
                ):
                    try:
                        request_state = inbound.get_nowait()
                        new_requests.append(request_state)
                    except Empty:
                        break

                for i, req in enumerate(new_requests):
                    if control.should_stop():
                        self._cancel_requests(
                            new_requests[i:] + self._active,
                            "Worker is shutting down, request cancelled",
                            callbacks.cancel_request,
                            "prefill",
                        )
                        self._active.clear()
                        return

                    try:
                        result = self._executor.prefill(req)
                    except Exception as e:
                        callbacks.handle_fatal_error(e, new_requests[i:])
                        return
                    if isinstance(result, RequestFailure):
                        self._emitter.on_failed(req, result.error)
                    elif isinstance(result, PrefillResult):
                        self._emitter.on_prefill_started(req, result.start_ns)
                        self._emitter.on_prefill_succeeded(req, result)
                    if req.status == RequestStatus.DECODING:
                        self._active.append(req)

                for req in self._active:
                    if control.should_stop():
                        self._cancel_requests(
                            self._active,
                            "Worker is shutting down, request cancelled",
                            callbacks.cancel_request,
                            "decoding",
                        )
                        self._active.clear()
                        return

                    if req.status == RequestStatus.DECODING:
                        result = self._executor.decode(req)
                        if isinstance(result, RequestFailure):
                            self._emitter.on_failed(req, result.error)
                        elif isinstance(result, DecodeResult):
                            self._emitter.on_token(req, result)

                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                if not self._active:
                    control.wait_idle(0.01)
        except Exception as e:
            callbacks.handle_fatal_error(e, None)


class BatchInferenceEngine:
    def __init__(
        self, executor: BaseBatchExecutor, config: BatchExecutorConfig
    ) -> None:
        self._executor = executor
        self._config = config
        self._waiting: list[GenerationRequestState] = []
        self._active: list[GenerationRequestState] = []
        self._emitter = RequestEventEmitter()

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        for pending in self._waiting + self._active:
            try:
                cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for request %s",
                    pending.request_id,
                )
        self._waiting.clear()
        self._active.clear()

    def drain_inbound(self, inbound: Queue[GenerationRequestState]) -> None:
        max_num_reqs = self._config.max_active_requests
        while len(self._waiting) + len(self._active) < max_num_reqs:
            try:
                req = inbound.get_nowait()
                self._waiting.append(req)
            except Empty:
                break

    def select_prefill_batch(self) -> list[GenerationRequestState]:
        batch_size = min(
            self._config.max_prefill_batch_size,
            self._config.max_active_requests - len(self._active),
            len(self._waiting),
        )
        batch = self._waiting[:batch_size]
        self._waiting = self._waiting[batch_size:]
        return batch

    def select_decode_batch(self) -> list[GenerationRequestState]:
        decoding = [req for req in self._active if req.status == RequestStatus.DECODING]
        return decoding[: self._config.max_decode_batch_size]

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        prefill_batch: list[GenerationRequestState] = []
        try:
            while not control.should_stop():
                self.drain_inbound(inbound)

                if not control.should_stop():
                    prefill_batch = self.select_prefill_batch()
                    if prefill_batch:
                        results = self._executor.batched_prefill(prefill_batch)
                        if len(results) != len(prefill_batch):
                            raise ValueError(
                                f"Expected {len(prefill_batch)} prefill results, but got {len(results)}"
                            )
                        for req, result in zip(prefill_batch, results):
                            if isinstance(result, RequestFailure):
                                self._emitter.on_failed(req, result.error)
                            elif isinstance(result, PrefillResult):
                                self._emitter.on_prefill_started(req, result.start_ns)
                                self._emitter.on_prefill_succeeded(req, result)
                        self._active.extend(
                            req
                            for req in prefill_batch
                            if req.status == RequestStatus.DECODING
                        )
                        prefill_batch = []

                if not control.should_stop():
                    decoding_batch = self.select_decode_batch()
                    if decoding_batch:
                        results = self._executor.batched_decode(decoding_batch)
                        if len(results) != len(decoding_batch):
                            raise ValueError(
                                f"Expected {len(decoding_batch)} decode results, but got {len(results)}"
                            )
                        for req, result in zip(decoding_batch, results):
                            if isinstance(result, RequestFailure):
                                self._emitter.on_failed(req, result.error)
                            elif isinstance(result, DecodeResult):
                                self._emitter.on_token(req, result)

                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                if not self._waiting and not self._active:
                    control.wait_idle(0.01)
        except Exception as e:
            callbacks.handle_fatal_error(e, prefill_batch)
