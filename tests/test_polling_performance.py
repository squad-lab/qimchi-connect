"""
Deterministic complexity guardrails for repeated live polling.

A cache miss necessarily reads and frames the complete dataset until the
revisioned-delta protocol exists. Once that result is cached, however, another
poll must reuse the payload and binary blob without repeating dataset-sized
work. These tests count those operations instead of measuring wall-clock time.

"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import server
from qimchi_connect.protocol import PROTOCOL_VERSION


@pytest.mark.parametrize("points", [64, 1_048_576])
def test_cached_polls_reuse_one_dataset_sized_build(
    points: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many cache hits cost one provider call and one serialization in total."""
    dataset = xr.Dataset(
        {"signal": ("point", np.arange(points, dtype=np.float64))},
        coords={"point": np.arange(points, dtype=np.int64)},
    )
    provider_calls = 0
    build_calls = 0

    def provider() -> xr.Dataset:
        nonlocal provider_calls
        provider_calls += 1
        return dataset

    original_build_snapshot = server._build_snapshot

    def counted_build_snapshot(measurement_id: str, snapshot_provider):
        nonlocal build_calls
        build_calls += 1
        return original_build_snapshot(measurement_id, snapshot_provider)

    monkeypatch.setattr(server, "SNAPSHOT_CACHE_TTL", float("inf"))
    monkeypatch.setattr(server, "_build_snapshot", counted_build_snapshot)
    server.register_snapshot_provider("complexity", provider)

    async def poll_repeatedly() -> list[dict]:
        request = {
            "action": "get_snapshot",
            "measurement_id": "complexity",
            "protocol_version": PROTOCOL_VERSION,
        }
        return [await server._process_request(request) for _ in range(32)]

    responses = asyncio.run(poll_repeatedly())

    assert provider_calls == 1
    assert build_calls == 1
    first_blob = responses[0]["_binary_payload"]
    assert all(response["_binary_payload"] is first_blob for response in responses)
