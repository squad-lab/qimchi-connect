import asyncio

import xarray as xr

from qimchi_connect import client, server
from qimchi_connect.protocol import PROTOCOL_NAME, PROTOCOL_VERSION


def test_every_response_identifies_the_protocol():
    dataset = xr.Dataset({"signal": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    server.register_snapshot_provider("run-1", lambda: dataset)

    snapshot = asyncio.run(
        server._process_request({"action": "get_snapshot", "measurement_id": "run-1"})
    )
    listing = asyncio.run(server._process_request({"action": "list"}))
    failure = asyncio.run(server._process_request({"action": "bogus"}))

    for response in (snapshot, listing, failure):
        assert response["protocol"] == PROTOCOL_NAME
        assert response["protocol_version"] == PROTOCOL_VERSION
    assert snapshot["measurement_id"] == "run-1"


def test_a_snapshot_carries_full_variable_metadata():
    dataset = xr.Dataset({"signal": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    server.register_snapshot_provider("run-1", lambda: dataset)

    response = asyncio.run(
        server._process_request({"action": "get_snapshot", "measurement_id": "run-1"})
    )

    assert set(response["var_dims"]) == {"x", "signal"}
    assert set(response["var_attrs"]) == {"x", "signal"}
    assert set(response["var_dtypes"]) == {"x", "signal"}
    assert response["_binary_payload"]


def test_an_empty_server_lists_nothing_rather_than_failing():
    response = asyncio.run(server._process_request({"action": "list"}))

    assert response["success"] is True
    assert response["measurements"] == []


def test_background_server_serves_registered_measurement():
    dataset = xr.Dataset({"signal": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    server.register_snapshot_provider("run-2", lambda: dataset)

    assert server.start_live_server(port=0)

    endpoint = f"ws://localhost:{server.get_server_port()}"
    restored = client.open_live_measurement_sync("run-2", endpoint)

    xr.testing.assert_identical(restored, dataset)
