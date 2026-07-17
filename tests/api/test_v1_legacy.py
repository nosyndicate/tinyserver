"""Mounting contract for the preserved v1 baseline (see server/api/v1_legacy.py).

v1 endpoints are an explicit opt-in mode: they exist only when the server is
started with ``--api-version v1`` and never ride along with the queue-based
versions. These tests pin that contract via ``create_app`` route inspection —
no lifespan is run, so no model/CUDA is needed.
"""

import argparse

import pytest
from fastapi import FastAPI

# server.main transitively imports the triton kernels via the v4 stack, which
# aren't installed on non-GPU dev machines — skip there, run on the GPU box.
main = pytest.importorskip("server.main")
create_app = main.create_app

V1_PATHS = {"/generate", "/generate/stream"}


def make_app(api_version: str) -> FastAPI:
    return create_app(argparse.Namespace(api_version=api_version))


def route_paths(app: FastAPI) -> set[str]:
    return {route.path for route in app.routes}


def test_v1_mode_mounts_only_v1_and_health() -> None:
    paths = route_paths(make_app("v1"))
    assert V1_PATHS <= paths
    assert "/health" in paths
    # No queue-based endpoints in v1 mode.
    assert not any(p.endswith(("_v2", "_v3", "_v4")) for p in paths)


def test_v2_and_v3_modes_do_not_mount_v1() -> None:
    for version in ("v2", "v3"):
        paths = route_paths(make_app(version))
        assert V1_PATHS.isdisjoint(paths), f"{version} mode must not serve v1"
        assert f"/generate_{version}" in paths
        assert f"/generate/stream_{version}" in paths


def test_v4_mode_unchanged_no_v1() -> None:
    paths = route_paths(make_app("v4"))
    assert V1_PATHS.isdisjoint(paths)
    assert "/generate_v4" in paths
    assert "/generate/stream_v4" in paths
    assert "/health" in paths
