"""
Tests for concurrent live polling.

Several live plots of one measurement poll the server independently, and the
server shares a process with the measurement that produces the data. A slow
snapshot must not block the event loop, concurrent pollers share one snapshot
build, and each payload is converted once.

"""

import asyncio
import threading
import time

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import client, protocol, server


def _dataset(size: int = 8) -> xr.Dataset:
    """Build a small gridded dataset shaped like a live sweep."""
    return xr.Dataset(
        {"signal": (("x", "y"), np.random.rand(size, size))},
        coords={"x": np.arange(float(size)), "y": np.arange(float(size))},
    )


def test_slow_snapshot_does_not_block_the_event_loop():
    """A snapshot in progress must not stall unrelated requests or handshakes."""
    dataset = _dataset()
    started = threading.Event()

    def slow_snapshot() -> xr.Dataset:
        started.set()
        time.sleep(1.0)
        return dataset

    server.register_snapshot_provider("slow", slow_snapshot)
    server.register_snapshot_provider("fast", lambda: dataset)
    assert server.start_live_server(port=0)
    endpoint = f"ws://localhost:{server.get_server_port()}"

    async def scenario() -> float:
        slow = asyncio.create_task(client.open_live_measurement("slow", endpoint))
        await asyncio.to_thread(started.wait, 5.0)
        start = time.monotonic()
        await client.open_live_measurement("fast", endpoint)
        elapsed = time.monotonic() - start
        await slow
        return elapsed

    # The fast request must not be serialized behind the slow snapshot.
    assert asyncio.run(scenario()) < 0.5


def test_concurrent_polls_share_one_snapshot_build():
    """Pollers arriving together pay for a single serialization, not one each."""
    dataset = _dataset()
    builds = 0

    def counting_snapshot() -> xr.Dataset:
        nonlocal builds
        builds += 1
        time.sleep(0.05)
        return dataset

    server.register_snapshot_provider("shared", counting_snapshot)
    assert server.start_live_server(port=0)
    endpoint = f"ws://localhost:{server.get_server_port()}"

    async def poll_together() -> None:
        await asyncio.gather(
            *[client.open_live_measurement("shared", endpoint) for _ in range(6)]
        )

    asyncio.run(poll_together())

    assert builds == 1


def test_snapshot_cache_expires():
    """A cached snapshot must not outlive its TTL, or live plots would freeze."""
    builds = 0

    def counting_snapshot() -> xr.Dataset:
        nonlocal builds
        builds += 1
        return _dataset()

    server.register_snapshot_provider("expiring", counting_snapshot)

    async def poll_twice() -> None:
        await server._process_request(
            {"action": "get_snapshot", "measurement_id": "expiring"}
        )
        await asyncio.sleep(server.SNAPSHOT_CACHE_TTL + 0.05)
        await server._process_request(
            {"action": "get_snapshot", "measurement_id": "expiring"}
        )

    asyncio.run(poll_twice())

    assert builds == 2


def test_serving_a_snapshot_does_not_mutate_the_shared_cache_entry():
    """Every caller of one cache entry must see the full field set."""
    server.register_snapshot_provider("mixed", _dataset)

    async def fetch_twice() -> dict:
        await server._process_request(
            {"action": "get_snapshot", "measurement_id": "mixed"}
        )
        return await server._process_request(
            {
                "action": "get_snapshot",
                "measurement_id": "mixed",
                "protocol_version": protocol.PROTOCOL_VERSION,
            }
        )

    current = asyncio.run(fetch_twice())

    assert "var_attrs" in current
    assert "var_dtypes" in current
    assert set(current["var_dims"]) == {"x", "y", "signal"}


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.5, np.nan, np.inf]),
        np.array([1, 2, 3], dtype=np.int32),
        np.array([True, False]),
        np.array([1 + 2j, 3 - 4j]),
        np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
        np.array([b"\x00\x01", b"\x02"], dtype=object),
    ],
)
def test_payloads_survive_a_single_conversion_pass(values):
    """
    Responses are encoded directly, so they must already be JSON-ready.
    Numeric and boolean dtypes take the binary path instead
    (protocol.pack_snapshot).

    """
    dataset = xr.Dataset({"v": ("i", values)}, coords={"i": np.arange(len(values))})
    server.register_snapshot_provider("dtypes", lambda: dataset)

    response = asyncio.run(
        server._process_request(
            {
                "action": "get_snapshot",
                "measurement_id": "dtypes",
                "protocol_version": protocol.PROTOCOL_VERSION,
            }
        )
    )

    # A successful get_snapshot response is always binary-
    # framed: the coordinate `i` is int64, so it takes the binary path in every
    # case regardless of `v`'s dtype. What varies is whether `v` itself rides
    # the binary payload -- only plain numeric and bool dtypes do; complex,
    # datetime and object-dtype arrays stay JSON-encoded within the same
    # message (see snapshot_payload).
    binary_payload = response.get("_binary_payload")
    assert binary_payload is not None
    assert (values.dtype.kind in "biuf") == ("v" in response.get("binary_vars", {}))

    # unpack_snapshot re-derives array values only for the binary_vars it
    # finds; everything else -- including json_default's complex, datetime and
    # bytes encodings -- passes through the header as json_compatible left it.
    # One call therefore exercises both decode paths, as a real client does.
    restored = protocol.measurement_from_snapshot(
        protocol.unpack_snapshot(binary_payload)
    )

    assert str(restored["v"].dtype) == str(values.dtype)
    if values.dtype.kind == "f":
        np.testing.assert_array_equal(restored["v"].values, values)
    else:
        assert list(restored["v"].values) == list(values)


def test_connect_and_response_budgets_are_separate():
    """A slow answer must not be charged against the connection budget."""
    dataset = _dataset()

    def slow_snapshot() -> xr.Dataset:
        time.sleep(1.0)
        return dataset

    server.register_snapshot_provider("patient", slow_snapshot)
    assert server.start_live_server(port=0)
    endpoint = f"ws://localhost:{server.get_server_port()}"

    # A connect budget far shorter than the snapshot still succeeds, because it
    # only covers reaching the server.
    restored = asyncio.run(
        client.open_live_measurement(
            "patient", endpoint, timeout=10.0, connect_timeout=0.5
        )
    )

    xr.testing.assert_identical(restored, dataset)


def test_response_budget_still_bounds_a_hung_server():
    """A separate connection budget must not remove the ceiling on a poll."""
    server.register_snapshot_provider("stuck", lambda: time.sleep(30) or _dataset())
    assert server.start_live_server(port=0)
    endpoint = f"ws://localhost:{server.get_server_port()}"

    with pytest.raises(TimeoutError):
        asyncio.run(client.open_live_measurement("stuck", endpoint, timeout=0.5))
