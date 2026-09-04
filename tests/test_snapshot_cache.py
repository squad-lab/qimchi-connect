"""
Tests for snapshot cache lifetime and invalidation.

The server keeps one serialized snapshot per measurement for SNAPSHOT_CACHE_TTL
so that several live plots of one measurement share a single build. The edges
of that are when a provider is replaced, when it is removed, and when a build
finishes after either.

"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import xarray as xr

from qimchi_connect import server
from qimchi_connect.protocol import (
    PROTOCOL_VERSION,
    measurement_from_snapshot,
    unpack_snapshot,
)


def _sizes(response: dict) -> list[float]:
    restored = measurement_from_snapshot(unpack_snapshot(response["_binary_payload"]))
    return restored["a"].values.tolist()


def _snapshot(measurement_id: str) -> dict:
    return asyncio.run(
        server._process_request(
            {
                "action": "get_snapshot",
                "measurement_id": measurement_id,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
    )


class TestInvalidation:
    def test_replacing_a_provider_is_visible_immediately(self):
        """
        A measurement re-registered under the same id inside the TTL must not be
        answered from the previous provider's snapshot.

        """
        server.register_snapshot_provider(
            "run", lambda: xr.Dataset({"a": ("i", [1.0])})
        )
        first = _snapshot("run")

        server.register_snapshot_provider(
            "run", lambda: xr.Dataset({"a": ("i", [2.0, 3.0, 4.0])})
        )
        second = _snapshot("run")

        assert _sizes(first) == [1.0]
        assert _sizes(second) == [2.0, 3.0, 4.0]

    def test_a_close_and_reopen_cycle_is_visible_immediately(self):
        server.register_snapshot_provider(
            "run", lambda: xr.Dataset({"a": ("i", [1.0])})
        )
        _snapshot("run")
        server.unregister_snapshot_provider("run")

        server.register_snapshot_provider(
            "run", lambda: xr.Dataset({"a": ("i", [7.0])})
        )

        assert _sizes(_snapshot("run")) == [7.0]

    def test_an_unrelated_measurement_is_still_served(self):
        server.register_snapshot_provider("a", lambda: xr.Dataset({"a": ("i", [1.0])}))
        server.register_snapshot_provider("b", lambda: xr.Dataset({"a": ("i", [2.0])}))

        _snapshot("a")
        server.unregister_snapshot_provider("a")

        assert _sizes(_snapshot("b")) == [2.0]


class TestRetention:
    def test_closing_a_measurement_releases_its_cached_snapshot(self):
        """
        A producer process runs many measurements in a row, so a snapshot left
        behind by each closed measurement accumulates for the process's lifetime.

        """
        server.register_snapshot_provider(
            "big", lambda: xr.Dataset({"a": ("i", np.zeros(10_000))})
        )
        _snapshot("big")
        assert "big" in server._SNAPSHOT_CACHE

        server.unregister_snapshot_provider("big")

        assert "big" not in server._SNAPSHOT_CACHE
        assert "big" not in server._SNAPSHOT_LOCKS

    def test_a_build_finishing_after_a_close_is_not_cached(self):
        """The result is still returned to its caller; it just is not kept."""
        entered = threading.Event()
        release = threading.Event()

        def slow() -> xr.Dataset:
            entered.set()
            release.wait(5.0)
            return xr.Dataset({"a": ("i", [1.0])})

        server.register_snapshot_provider("racy", slow)

        async def scenario() -> None:
            request = asyncio.create_task(
                server._process_request(
                    {
                        "action": "get_snapshot",
                        "measurement_id": "racy",
                        "protocol_version": PROTOCOL_VERSION,
                    }
                )
            )
            await asyncio.to_thread(entered.wait, 5.0)
            server.unregister_snapshot_provider("racy")
            release.set()
            response = await request
            assert _sizes(response) == [1.0]

        asyncio.run(scenario())

        assert "racy" not in server._SNAPSHOT_CACHE

    def test_stopping_the_server_clears_everything(self):
        server.register_snapshot_provider("x", lambda: xr.Dataset({"a": ("i", [1.0])}))
        _snapshot("x")

        server.stop_live_server()

        assert server._SNAPSHOT_CACHE == {}
        assert server._SNAPSHOT_LOCKS == {}
