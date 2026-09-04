"""
Tests for the request actions other than get_snapshot.

``get_data`` and ``get_zarr_array`` are what a consumer uses to inspect a
measurement's structure, or to pull a single variable, without transferring
the whole thing. Both the wire path (through the client helpers) and the
response shaping (through ``_process_request``) matter.

"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import client, server
from qimchi_connect.protocol import PROTOCOL_VERSION


@pytest.fixture
def dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "signal": (("x",), np.arange(4.0), {"unit": "V"}),
            "other": (("x",), np.arange(4.0) * 2),
        },
        coords={"x": np.arange(4)},
        attrs={"Measurement ID": "run-1"},
    )


@pytest.fixture
def published(dataset: xr.Dataset) -> xr.Dataset:
    """Register the measurement without starting a server, for in-process calls."""
    server.register_snapshot_provider(
        "run-1", lambda: dataset, {"source_package": "test"}
    )
    return dataset


@pytest.fixture
def endpoint(published: xr.Dataset) -> str:
    assert server.start_live_server(port=0)
    return f"ws://localhost:{server.get_server_port()}"


class TestCapabilities:
    def test_it_advertises_every_action_the_server_handles(self):
        response = asyncio.run(server._process_request({"action": "capabilities"}))

        assert response["success"] is True
        assert set(response["actions"]) == {
            "capabilities",
            "list",
            "get_data",
            "get_snapshot",
            "get_zarr_array",
        }

    def test_it_names_the_framing_and_the_id_field(self):
        """
        A client implementing the protocol from `capabilities` has to be told
        that a snapshot arrives as a binary frame, and what to call the id.

        """
        response = asyncio.run(server._process_request({"action": "capabilities"}))

        assert response["snapshot_format"] == "xarray-binary"
        assert response["id_field"] == "measurement_id"
        assert "legacy_snapshot_format" not in response


class TestGetData:
    def test_without_variables_it_returns_structure_and_attrs(self, endpoint):
        info = asyncio.run(client.get_measurement_info("run-1", endpoint))

        assert info["coords"] == ["x"]
        assert set(info["data_vars"]) == {"signal", "other"}
        assert info["attrs"]["Measurement ID"] == "run-1"

    def test_the_structure_response_carries_producer_metadata(self, published):
        response = asyncio.run(
            server._process_request(
                {
                    "action": "get_data",
                    "measurement_id": "run-1",
                    "protocol_version": PROTOCOL_VERSION,
                }
            )
        )

        assert response["source"] == {"source_package": "test"}

    def test_with_variables_it_returns_only_those_values(self, endpoint, dataset):
        data = asyncio.run(client.get_measurement_data("run-1", ["signal"], endpoint))

        assert set(data) == {"signal"}
        np.testing.assert_array_equal(np.asarray(data["signal"]), dataset["signal"])

    def test_a_coordinate_can_be_fetched_like_any_variable(self, endpoint):
        data = asyncio.run(client.get_measurement_data("run-1", ["x"], endpoint))

        assert data["x"] == [0, 1, 2, 3]

    def test_unknown_names_are_dropped_rather_than_failing(self, endpoint):
        data = asyncio.run(
            client.get_measurement_data("run-1", ["signal", "nope"], endpoint)
        )

        assert set(data) == {"signal"}

    def test_a_variable_request_omits_the_producer_metadata(self, published):
        """``source`` describes the measurement, so it rides the structure
        response rather than every value fetch."""
        response = asyncio.run(
            server._process_request(
                {
                    "action": "get_data",
                    "measurement_id": "run-1",
                    "variables": ["signal"],
                }
            )
        )

        assert response["measurement_id"] == "run-1"
        assert response["protocol_version"] == PROTOCOL_VERSION
        assert "source" not in response


class TestGetZarrArray:
    def test_it_returns_one_variable_with_its_shape_and_dtype(self, published):
        response = asyncio.run(
            server._process_request(
                {
                    "action": "get_zarr_array",
                    "measurement_id": "run-1",
                    "array_path": "signal",
                }
            )
        )

        assert response["success"] is True
        assert response["shape"] == [4]
        assert response["dtype"] == "float64"
        np.testing.assert_array_equal(
            np.asarray(response["data"]), published["signal"].values
        )

    def test_it_reports_the_variable_dims_and_attrs(self, published):
        response = asyncio.run(
            server._process_request(
                {
                    "action": "get_zarr_array",
                    "array_path": "signal",
                    "measurement_id": "run-1",
                }
            )
        )

        assert response["dims"] == ["x"]
        assert response["attrs"] == {"unit": "V"}

    def test_a_missing_variable_is_an_error_not_an_empty_result(self, published):
        response = asyncio.run(
            server._process_request(
                {
                    "action": "get_zarr_array",
                    "measurement_id": "run-1",
                    "array_path": "nope",
                }
            )
        )

        assert response["success"] is False
        assert "nope" in response["error"]

    def test_an_empty_array_path_is_rejected(self, published):
        response = asyncio.run(
            server._process_request(
                {
                    "action": "get_zarr_array",
                    "measurement_id": "run-1",
                    "array_path": "",
                }
            )
        )

        assert response["success"] is False


class TestErrorResponses:
    def test_a_request_without_an_identifier_says_so(self):
        response = asyncio.run(server._process_request({"action": "get_snapshot"}))

        assert response["success"] is False
        assert response["error"] == "measurement_id required"

    def test_an_unknown_measurement_is_reported_by_name(self):
        response = asyncio.run(
            server._process_request(
                {"action": "get_snapshot", "measurement_id": "absent"}
            )
        )

        assert response["success"] is False
        assert "absent" in response["error"]

    def test_an_unknown_action_is_reported_by_name(self, published):
        response = asyncio.run(
            server._process_request({"action": "bogus", "measurement_id": "run-1"})
        )

        assert response["success"] is False
        assert "bogus" in response["error"]

    def test_a_provider_that_raises_becomes_an_error_response(self):
        def broken() -> xr.Dataset:
            raise RuntimeError("instrument on fire")

        server.register_snapshot_provider("broken", broken)

        response = asyncio.run(
            server._process_request(
                {"action": "get_snapshot", "measurement_id": "broken"}
            )
        )

        assert response["success"] is False
        assert "instrument on fire" in response["error"]

    def test_a_provider_returning_the_wrong_type_is_rejected(self):
        server.register_snapshot_provider("wrong", lambda: 42)

        response = asyncio.run(
            server._process_request(
                {"action": "get_snapshot", "measurement_id": "wrong"}
            )
        )

        assert response["success"] is False
        assert "expected xarray.Dataset" in response["error"]

    def test_registration_rejects_an_empty_identifier(self):
        with pytest.raises(ValueError):
            server.register_snapshot_provider("  ", lambda: xr.Dataset())

    def test_registration_rejects_a_non_callable_snapshot(self):
        with pytest.raises(TypeError):
            server.register_snapshot_provider("x", xr.Dataset())


class TestConnectionHandling:
    def test_a_malformed_request_gets_an_error_rather_than_a_dropped_connection(
        self, endpoint
    ):
        import json

        import websockets

        async def send_garbage() -> dict:
            async with websockets.connect(endpoint) as socket:
                await socket.send("this is not json")
                return json.loads(await socket.recv())

        assert asyncio.run(send_garbage()) == {
            "success": False,
            "error": "Invalid JSON",
        }

    def test_a_connection_serves_several_requests(self, endpoint):
        import json

        import websockets

        async def two_requests() -> list[dict]:
            async with websockets.connect(endpoint) as socket:
                responses = []
                for action in ("capabilities", "list"):
                    await socket.send(json.dumps({"action": action}))
                    responses.append(json.loads(await socket.recv()))
                return responses

        assert all(response["success"] for response in asyncio.run(two_requests()))

    def test_a_client_disconnecting_mid_session_does_not_disturb_the_server(
        self, endpoint
    ):
        import websockets

        async def connect_then_leave() -> None:
            socket = await websockets.connect(endpoint)
            await socket.close()

        asyncio.run(connect_then_leave())

        assert server.is_server_running()
        assert asyncio.run(server._process_request({"action": "capabilities"}))[
            "success"
        ]
