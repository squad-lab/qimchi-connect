"""
Tests for the binary snapshot wire format.

``snapshot_payload(dataset, binary=True)`` moves numeric and boolean arrays
out of the JSON body and into a raw byte blob (``pack_snapshot`` /
``unpack_snapshot``). The round trip is exact for every dtype qanary
produces, the framed message is smaller than the JSON one, and a truncated
message raises.

"""

import asyncio

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import client, server
from qimchi_connect.protocol import (
    measurement_from_snapshot,
    pack_snapshot,
    snapshot_payload,
    unpack_snapshot,
)


def _measurement_shaped_dataset(size: int = 40) -> xr.Dataset:
    """A dataset shaped like a live qanary measurement: 2D float grid, metadata."""
    return xr.Dataset(
        {"signal": (("x", "y"), np.random.default_rng(0).random((size, size)))},
        coords={
            "x": np.linspace(0.0, 1.0, size),
            "y": np.linspace(0.0, 1.0, size),
        },
        attrs={"Sweeps": "{}", "Measurement ID": "test-1"},
    )


class TestBinaryRoundTrip:
    """``pack_snapshot`` -> ``unpack_snapshot`` must reconstruct exactly."""

    def test_a_realistic_measurement_grid_round_trips_exactly(self):
        dataset = _measurement_shaped_dataset()

        payload = snapshot_payload(dataset, binary=True)
        raw = pack_snapshot(payload, dataset)
        restored = measurement_from_snapshot(unpack_snapshot(raw))

        xr.testing.assert_identical(restored, dataset)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.5, np.nan, np.inf, -np.inf, 0.0]),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([1.5, 2.5], dtype=np.float32),
            np.array([True, False, True]),
        ],
    )
    def test_every_binary_eligible_dtype_round_trips_exactly(self, values):
        dataset = xr.Dataset({"v": ("i", values)}, coords={"i": np.arange(len(values))})

        payload = snapshot_payload(dataset, binary=True)
        assert "v" in payload["binary_vars"], "this dtype should take the binary path"

        raw = pack_snapshot(payload, dataset)
        restored = measurement_from_snapshot(unpack_snapshot(raw))

        assert str(restored["v"].dtype) == str(values.dtype)
        if values.dtype.kind == "f":
            np.testing.assert_array_equal(restored["v"].values, values, strict=True)
        else:
            np.testing.assert_array_equal(restored["v"].values, values)

    def test_a_measurement_with_no_binary_eligible_variables_still_round_trips(self):
        """An all-string/complex dataset: binary_vars is empty, blob is empty."""
        dataset = xr.Dataset(
            {"label": ("i", np.array(["a", "bb"], dtype=object))},
            coords={"i": np.array(["x", "y"], dtype=object)},
        )

        payload = snapshot_payload(dataset, binary=True)
        assert payload["binary_vars"] == {}

        raw = pack_snapshot(payload, dataset)
        restored = measurement_from_snapshot(unpack_snapshot(raw))

        xr.testing.assert_identical(restored, dataset)


class TestSmallerOnTheWire:
    """The framed message must be smaller than the all-JSON payload."""

    def test_binary_framing_is_smaller_than_the_all_json_payload(self):
        import json

        from qimchi_connect.protocol import json_default

        dataset = _measurement_shaped_dataset(size=100)

        json_only = snapshot_payload(dataset, binary=False)
        json_bytes = len(json.dumps(json_only, default=json_default).encode("utf-8"))

        binary_payload = snapshot_payload(dataset, binary=True)
        binary_bytes = len(pack_snapshot(binary_payload, dataset))

        assert binary_bytes < json_bytes / 2


class TestServerResponses:
    """get_snapshot always answers with the binary framing."""

    def test_the_numeric_variables_ride_the_binary_payload(self):
        dataset = _measurement_shaped_dataset(size=8)
        server.register_snapshot_provider("target", lambda: dataset)

        response = asyncio.run(
            server._process_request(
                {"action": "get_snapshot", "measurement_id": "target"}
            )
        )

        assert set(response["binary_vars"]) == {"x", "y", "signal"}
        assert response["data"] == {}
        restored = measurement_from_snapshot(
            unpack_snapshot(response["_binary_payload"])
        )
        np.testing.assert_array_equal(
            restored["signal"].values, dataset["signal"].values
        )


class TestEndToEndOverTheRealSocket:
    """open_live_measurement over a real connection, as a consumer uses it."""

    def test_open_live_measurement_reconstructs_a_large_grid_exactly(self):
        dataset = _measurement_shaped_dataset(size=200)
        server.register_snapshot_provider("wire-target", lambda: dataset)
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        restored = client.open_live_measurement_sync("wire-target", endpoint)

        xr.testing.assert_identical(restored, dataset)

    def test_send_request_decodes_a_hand_built_snapshot_request(self):
        """send_request's binary/text dispatch must not depend on the action."""
        dataset = _measurement_shaped_dataset(size=8)
        server.register_snapshot_provider("raw-wire", lambda: dataset)
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        response = asyncio.run(
            client.send_request(
                {"action": "get_snapshot", "measurement_id": "raw-wire"},
                endpoint,
            )
        )

        assert response["success"] is True
        restored = measurement_from_snapshot(response)
        xr.testing.assert_identical(restored, dataset)


class TestTruncationIsDetected:
    """A cut-off binary message must raise, not silently misread the blob."""

    def test_a_message_shorter_than_its_declared_arrays_raises(self):
        dataset = xr.Dataset(
            {"v": ("i", np.arange(100, dtype=np.int64))}, coords={"i": np.arange(100)}
        )
        payload = snapshot_payload(dataset, binary=True)
        raw = pack_snapshot(payload, dataset)

        with pytest.raises(ValueError, match="truncated"):
            unpack_snapshot(raw[:-50])
