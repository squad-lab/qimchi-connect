import numpy as np
import xarray as xr

from qimchi_connect.protocol import measurement_from_snapshot, snapshot_payload


def test_snapshot_round_trip_preserves_xarray_structure_and_metadata():
    dataset = xr.Dataset(
        data_vars={
            "signal": (
                ("row", "column"),
                np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]),
                {"unit": "V"},
            )
        },
        coords={
            "row": ("row", [10, 20], {"axis": "slow"}),
            "column": ("column", [1.5, 2.5]),
            "label": ((), "sample-a"),
        },
        attrs={"package": "test"},
    )

    restored = measurement_from_snapshot(snapshot_payload(dataset))

    xr.testing.assert_identical(restored, dataset)
