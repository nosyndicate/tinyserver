"""Argument contract for the server entrypoint (see server/main.py).

Each API version is a subcommand, so a version can only be given the flags it
actually reads. That structure is the contract: previously every flag lived on
one flat parser, and a version that ignored a flag accepted it silently — a
benchmark sweep could appear to vary a knob the running version never consults,
which is the same class of defect the benchmark measurement correction exists to
remove.

Three properties are pinned here: every default reproduces the behavior that was
previously hardcoded, a version rejects flags belonging to another version, and
an impossible batch combination is rejected at parse time rather than after the
model is loaded.

No lifespan runs, so no model or CUDA is needed.
"""

import argparse

import pytest
from fastapi import FastAPI

# server.main transitively imports the triton kernels via the v4 stack, which
# aren't installed on non-GPU dev machines — skip there, run on the GPU box.
main = pytest.importorskip("server.main")
parse_args = main.parse_args
create_app = main.create_app

from server.executor.engine import validate_batch_engine_config  # noqa: E402
from server.executor.types import BatchEngineConfig  # noqa: E402
from server.model.types import ModelConfig  # noqa: E402


class TestVersionIsRequired:
    def test_bare_invocation_exits(self) -> None:
        """The version under test is the whole point; never imply one."""
        with pytest.raises(SystemExit) as excinfo:
            parse_args([])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
    def test_each_version_parses(self, version: str) -> None:
        assert parse_args([version]).api_version == version


class TestFlagScoping:
    """A version rejects flags it does not read, rather than ignoring them."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["v1", "--worker-queue-size", "128"],  # v1 has no worker
            ["v1", "--max-active-requests", "8"],
            ["v2", "--max-prefill-batch-size", "4"],  # batch sizes are v3-only
            ["v2", "--max-decode-batch-size", "4"],
            ["v2", "--max-num-sequences", "8"],  # v4-only scheduler knob
            ["v3", "--max-num-sequences", "8"],
            ["v3", "--memory-utilization", "0.9"],  # v4-only KV pool knob
            ["v3", "--block-size", "128"],
            ["v4", "--max-active-requests", "8"],  # v4 takes no EngineConfig
            ["v4", "--max-prefill-batch-size", "4"],
        ],
    )
    def test_rejects_another_versions_flag(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            parse_args(argv)
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
    def test_shared_flags_follow_the_subcommand(self, version: str) -> None:
        args = parse_args([version, "--host", "127.0.0.1", "--port", "8123"])
        assert (args.host, args.port) == ("127.0.0.1", 8123)

    @pytest.mark.parametrize("version", ["v2", "v3", "v4"])
    def test_worker_queue_size_is_shared_by_the_queued_versions(
        self, version: str
    ) -> None:
        assert (
            parse_args([version, "--worker-queue-size", "256"]).worker_queue_size == 256
        )


class TestDefaults:
    """Every default reproduces a value that used to be hardcoded."""

    @pytest.mark.parametrize("version", ["v2", "v3", "v4"])
    def test_worker_queue_size_matches_the_old_hardcoded_value(
        self, version: str
    ) -> None:
        assert parse_args([version]).worker_queue_size == 64

    def test_v3_engine_defaults_match_the_dataclass_defaults(self) -> None:
        args = parse_args(["v3"])
        assert args.max_active_requests == 16
        assert args.max_prefill_batch_size == 8
        assert args.max_decode_batch_size == 8
        # The config built from defaults equals the no-argument dataclass,
        # which is what _build_worker used to construct directly.
        assert main._batch_engine_config(args) == BatchEngineConfig()

    def test_v2_engine_default_matches_the_dataclass_default(self) -> None:
        assert main._engine_config(parse_args(["v2"])).max_active_requests == 16

    def test_v4_scheduler_defaults(self) -> None:
        args = parse_args(["v4"])
        assert args.max_waiting == 64
        assert args.max_num_sequences == 8
        assert args.max_num_tokens == 4096

    @pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
    def test_bind_defaults_match_the_old_hardcoded_address(self, version: str) -> None:
        args = parse_args([version])
        assert (args.host, args.port) == ("0.0.0.0", 8000)


class TestFlagsReachConfigs:
    def test_batch_engine_config_carries_parsed_values(self) -> None:
        args = parse_args(
            [
                "v3",
                "--max-active-requests",
                "32",
                "--max-prefill-batch-size",
                "4",
                "--max-decode-batch-size",
                "16",
            ]
        )
        config = main._batch_engine_config(args)
        assert config.max_active_requests == 32
        assert config.max_prefill_batch_size == 4
        assert config.max_decode_batch_size == 16

    def test_engine_config_carries_parsed_value(self) -> None:
        # v2 has no batch-size flags, so a value below their v3 defaults is
        # legitimate here and cannot conflict with anything.
        args = parse_args(["v2", "--max-active-requests", "2"])
        assert main._engine_config(args).max_active_requests == 2


class TestModelConfig:
    """The KV-pool knobs exist only where a paged KV pool does."""

    def test_v4_carries_the_parsed_pool_settings(self) -> None:
        args = parse_args(["v4", "--block-size", "128", "--memory-utilization", "0.9"])
        config = main._model_config(args)
        assert config.block_size == 128
        assert config.memory_utilization == 0.9

    @pytest.mark.parametrize("version", ["v1", "v2", "v3"])
    def test_other_versions_take_the_dataclass_defaults(self, version: str) -> None:
        config = main._model_config(parse_args([version]))
        assert config.block_size == ModelConfig().block_size
        assert config.memory_utilization == ModelConfig().memory_utilization


class TestValidation:
    """The pure validator, tested without argparse in the way."""

    @pytest.mark.parametrize(
        "config, expected",
        [
            (
                BatchEngineConfig(max_active_requests=0),
                "max_active_requests must be positive",
            ),
            (
                BatchEngineConfig(max_prefill_batch_size=0),
                "max_prefill_batch_size must be positive",
            ),
            (
                BatchEngineConfig(max_decode_batch_size=-1),
                "max_decode_batch_size must be positive",
            ),
            (
                BatchEngineConfig(max_active_requests=8, max_prefill_batch_size=32),
                "max_prefill_batch_size cannot be greater than max_active_requests",
            ),
            (
                BatchEngineConfig(max_active_requests=8, max_decode_batch_size=32),
                "max_decode_batch_size cannot be greater than max_active_requests",
            ),
        ],
    )
    def test_rejects_unusable_configs(
        self, config: BatchEngineConfig, expected: str
    ) -> None:
        with pytest.raises(ValueError, match=expected):
            validate_batch_engine_config(config)

    def test_accepts_the_defaults(self) -> None:
        validate_batch_engine_config(BatchEngineConfig())


class TestParseTimeRejection:
    """The same failures, surfaced by argparse before any model is loaded."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["v3", "--max-prefill-batch-size", "32", "--max-active-requests", "8"],
            ["v3", "--max-decode-batch-size", "32", "--max-active-requests", "8"],
            ["v3", "--max-prefill-batch-size", "0"],
            ["v3", "--max-decode-batch-size", "-1"],
            ["v3", "--max-active-requests", "0"],
            ["v2", "--max-active-requests", "0"],
        ],
    )
    def test_invalid_combination_exits(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            parse_args(argv)
        assert excinfo.value.code == 2

    def test_valid_combination_parses(self) -> None:
        args = parse_args(
            ["v3", "--max-prefill-batch-size", "8", "--max-active-requests", "8"]
        )
        assert args.max_prefill_batch_size == 8


def route_paths(app: FastAPI) -> set[str]:
    return {route.path for route in app.routes}


def test_parsed_args_still_mount_routes() -> None:
    """A renamed flag must not silently break app construction."""
    assert "/generate_v3" in route_paths(create_app(parse_args(["v3"])))


def test_create_app_accepts_a_bare_namespace() -> None:
    """create_app reads only api_version; the tuning flags stay out of it."""
    app = create_app(argparse.Namespace(api_version="v2"))
    assert "/generate_v2" in route_paths(app)
