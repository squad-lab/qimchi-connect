"""
One connection per request.

``send_request`` opens a short-lived connection and closes it with the reply.
Sharing one socket between requests is not safe: two coroutines awaiting
``recv()`` on it raise ``cannot call recv while another coroutine is already
running recv``, and either one's error handling closes the socket out from
under the other. Qimchi polls several live plots at once, so requests do
overlap.

"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import client, server


def _dataset(size: int) -> xr.Dataset:
    return xr.Dataset(
        {"signal": ("x", np.arange(float(size)))}, coords={"x": np.arange(size)}
    )


@pytest.fixture
def endpoint() -> str:
    for name, size in (("alpha", 3), ("beta", 7)):
        server.register_snapshot_provider(name, lambda size=size: _dataset(size))
    assert server.start_live_server(port=0)
    return f"ws://localhost:{server.get_server_port()}"


@pytest.fixture
def connect_calls(monkeypatch) -> list:
    """Record every connection the client opens."""
    opened: list = []
    real_connect = client.websockets.connect

    def counting_connect(*args, **kwargs):
        opened.append(args[0] if args else kwargs.get("uri"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(client.websockets, "connect", counting_connect)
    return opened


def test_every_request_opens_its_own_connection(endpoint, connect_calls):
    """
    One connection per request is the property that keeps concurrent requests
    apart, so the count is what this asserts rather than the absence of an
    error.

    """

    async def scenario():
        await asyncio.gather(
            *[client.open_live_measurement("alpha", endpoint) for _ in range(8)]
        )

    asyncio.run(scenario())

    assert len(connect_calls) == 8


def test_concurrent_requests_all_succeed(endpoint):
    """Overlapping requests must not contend for a single socket."""

    async def scenario():
        return await asyncio.gather(
            *[client.open_live_measurement("alpha", endpoint) for _ in range(12)]
        )

    results = asyncio.run(scenario())

    assert len(results) == 12
    assert all(restored["signal"].size == 3 for restored in results)


def test_concurrent_requests_for_different_measurements_do_not_cross(endpoint):
    """A shared socket could also reply to the wrong waiter."""

    async def scenario():
        requests = []
        for _ in range(6):
            requests.append(client.open_live_measurement("alpha", endpoint))
            requests.append(client.open_live_measurement("beta", endpoint))
        return await asyncio.gather(*requests)

    results = asyncio.run(scenario())

    sizes = [restored["signal"].size for restored in results]
    assert sizes == [3, 7] * 6


def test_a_failing_request_does_not_disturb_its_neighbours(endpoint):
    """
    One request's failure must not close a socket another is using: on a
    shared connection that surfaces as ``ConnectionClosedOK`` in a request
    that had nothing wrong with it.

    """

    async def scenario():
        return await asyncio.gather(
            *[client.open_live_measurement("alpha", endpoint) for _ in range(4)],
            *[client.open_live_measurement("absent", endpoint) for _ in range(4)],
            return_exceptions=True,
        )

    results = asyncio.run(scenario())

    succeeded = [r for r in results if isinstance(r, xr.Dataset)]
    failed = [r for r in results if isinstance(r, Exception)]
    assert len(succeeded) == 4
    assert len(failed) == 4
    assert all("absent" in str(error) for error in failed)
    assert not any("recv" in str(error) for error in failed)


def test_mixed_actions_run_concurrently(endpoint):
    """Every action shares the same connection handling, not just snapshots."""

    async def scenario():
        return await asyncio.gather(
            client.list_live_measurements(endpoint),
            client.get_measurement_info("alpha", endpoint),
            client.open_live_measurement("beta", endpoint),
            client.get_measurement_data("alpha", ["signal"], endpoint),
            client.list_live_measurements(endpoint),
        )

    listed, info, restored, data, listed_again = asyncio.run(scenario())

    assert set(listed) == {"alpha", "beta"}
    assert listed_again == listed
    assert info["data_vars"] == ["signal"]
    assert restored["signal"].size == 7
    assert len(data["signal"]) == 3


def test_a_connection_is_not_reused_between_requests(endpoint, connect_calls):
    """Sequential requests each get their own connection too."""

    async def scenario():
        for _ in range(3):
            await client.open_live_measurement("alpha", endpoint)

    asyncio.run(scenario())

    assert len(connect_calls) == 3
