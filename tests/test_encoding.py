"""
Tests for the JSON encoding layer of the protocol.

Producer metadata and measurement attrs are arbitrary Python: qcutils writes Path
objects and timestamps, drivers write NumPy scalars, and complex amplitudes
appear in the data itself. ``json_compatible`` converts those ahead of time and
``json_default`` catches whatever reached the encoder unconverted, so both are
covered here alongside the markers that survive a round trip.

"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from qimchi_connect.protocol import (
    decode_json_value,
    json_compatible,
    json_default,
    measurement_from_snapshot,
    snapshot_payload,
)


class TestJsonCompatible:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.int64(3), 3),
            (np.float32(1.5), 1.5),
            (np.bool_(True), True),
            (np.array([1, 2]), [1, 2]),
            (np.array([[1.5], [2.5]]), [[1.5], [2.5]]),
            (Path("/tmp/run.nc"), str(Path("/tmp/run.nc"))),
            (datetime.date(2026, 1, 2), "2026-01-02"),
            (datetime.time(3, 4, 5), "03:04:05"),
            ({"a", "a"}, ["a"]),
            ((1, 2), [1, 2]),
        ],
    )
    def test_it_converts_what_the_encoder_cannot(self, value, expected):
        assert json_compatible(value) == expected

    def test_dictionary_keys_become_strings(self):
        assert json_compatible({1: "a", None: "b"}) == {"1": "a", "None": "b"}

    def test_it_recurses_through_nested_containers(self):
        nested = {"outer": [{"inner": np.array([1, 2])}]}

        assert json_compatible(nested) == {"outer": [{"inner": [1, 2]}]}

    def test_the_result_is_directly_encodable(self):
        """
        Responses are handed to json.dumps without a second conversion walk, so
        whatever json_compatible returns has to be encodable as it stands.

        """
        value = json_compatible(
            {
                "path": Path("/tmp/x"),
                "when": datetime.datetime(2026, 1, 2, 3, 4, 5),
                "amplitude": 1 + 2j,
                "raw": b"\x00\xff",
                "array": np.arange(3),
            }
        )

        assert json.loads(json.dumps(value)) == value


class TestMarkers:
    @pytest.mark.parametrize(
        "value", [1 + 2j, -3.5 - 0.25j, b"", b"\x00\x01\xff", complex(0, 0)]
    )
    def test_complex_and_bytes_survive_a_round_trip(self, value):
        encoded = json.loads(json.dumps(json_compatible(value)))

        assert decode_json_value(encoded) == value

    def test_markers_are_decoded_inside_nested_structures(self):
        encoded = json_compatible({"list": [1 + 2j], "map": {"raw": b"\x01"}})

        assert decode_json_value(encoded) == {"list": [1 + 2j], "map": {"raw": b"\x01"}}

    def test_a_dictionary_that_merely_resembles_a_marker_is_left_alone(self):
        value = {"__qimchi_connect_bytes__": "abc", "extra": 1}

        assert decode_json_value(value) == value


class TestJsonDefault:
    @pytest.mark.parametrize(
        "value",
        [
            np.int64(3),
            np.array([1, 2]),
            1 + 2j,
            b"\x01",
            Path("/tmp/x"),
            (1, 2),
            {1, 2},
        ],
    )
    def test_it_handles_what_the_encoder_hands_it(self, value):
        json.dumps(value, default=json_default)

    def test_a_datetime_becomes_an_iso_string(self):
        assert (
            json_default(datetime.datetime(2026, 1, 2, 3, 4, 5))
            == "2026-01-02T03:04:05"
        )

    def test_an_unsupported_type_is_a_type_error(self):
        class Opaque:
            pass

        with pytest.raises(TypeError, match="Opaque"):
            json_default(Opaque())

    def test_an_attr_the_producer_left_unconverted_still_encodes(self):
        """
        Attrs come from the producer, so a value json_compatible does not know
        about can still reach json.dumps.

        """
        dataset = xr.Dataset(
            {"v": ("i", [1.0])}, attrs={"when": datetime.date(2026, 1, 2)}
        )

        encoded = json.dumps(snapshot_payload(dataset), default=json_default)

        assert "2026-01-02" in encoded


class TestReconstruction:
    def test_a_variable_absent_from_data_is_skipped_rather_than_failing(self):
        """A truncated snapshot yields a smaller dataset, not an exception."""
        snapshot = {
            "coords": ["x", "missing_coord"],
            "data_vars": ["signal", "missing_var"],
            "var_dims": {"x": ["x"], "signal": ["x"]},
            "data": {"x": [0, 1], "signal": [1.0, 2.0]},
        }

        restored = measurement_from_snapshot(snapshot)

        assert list(restored.coords) == ["x"]
        assert list(restored.data_vars) == ["signal"]

    def test_a_value_that_does_not_fit_its_declared_dtype_is_decoded_anyway(self):
        snapshot = {
            "coords": [],
            "data_vars": ["label"],
            "var_dims": {"label": ["i"]},
            "var_dtypes": {"label": "float64"},
            "data": {"label": ["not-a-number"]},
        }

        restored = measurement_from_snapshot(snapshot)

        assert restored["label"].values.tolist() == ["not-a-number"]

    def test_an_empty_snapshot_yields_an_empty_dataset(self):
        assert len(measurement_from_snapshot({}).variables) == 0

    def test_variable_attrs_are_restored_onto_the_dataset(self):
        dataset = xr.Dataset(
            {"signal": (("x",), [1.0, 2.0], {"unit": "V", "gain": 1 + 0j})},
            coords={"x": ([0, 1])},
        )

        restored = measurement_from_snapshot(snapshot_payload(dataset))

        assert restored["signal"].attrs == {"unit": "V", "gain": 1 + 0j}
