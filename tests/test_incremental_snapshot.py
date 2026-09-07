"""
Tests for fetching only the rows a consumer is missing.

A sweep preallocates its grid and fills it row by row, so a poller that
re-fetches the whole measurement every second moves the same bytes over and
over -- at 16 M points that is 128 MB per refresh, whatever changed. A
consumer that says how many rows it already holds is answered with the rest,
and these pin the frontier arithmetic, the fallbacks that send everything
instead, and the size difference that motivates it.

"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import client, server
from qimchi_connect.protocol import append_dim, row_frontier, unpack_snapshot

ROWS, COLS = 8, 4


@pytest.fixture
def half_written() -> xr.Dataset:
    """A 8x4 grid whose first 5 rows have been measured."""
    values = np.full((ROWS, COLS), np.nan)
    values[:5] = np.arange(5 * COLS, dtype=float).reshape(5, COLS)
    return xr.Dataset(
        {"signal": (("y", "x"), values)},
        coords={"y": np.arange(ROWS), "x": np.arange(COLS)},
    )


@pytest.fixture
def published(half_written: xr.Dataset) -> xr.Dataset:
    server.register_snapshot_provider("run-1", lambda: half_written)
    return half_written


def _snapshot(**extra):
    request = {"action": "get_snapshot", "measurement_id": "run-1", **extra}
    return asyncio.run(server._process_request(request))


class TestFrontier:
    def test_the_unmeasured_tail_is_not_counted_as_written(self, half_written):
        assert append_dim(half_written) == "y"
        assert row_frontier(half_written, "y") == 5

    def test_a_dtype_without_nan_reports_every_row(self):
        ds = xr.Dataset({"count": (("y",), np.arange(4))}, coords={"y": np.arange(4)})

        assert row_frontier(ds, "y") == 4

    def test_variables_that_disagree_on_their_leading_dim_have_no_append_dim(self):
        ds = xr.Dataset(
            {"a": (("y",), np.zeros(3)), "b": (("x",), np.zeros(3))},
            coords={"y": np.arange(3), "x": np.arange(3)},
        )

        assert append_dim(ds) is None


class TestPartialSnapshot:
    def test_a_full_snapshot_advertises_where_the_writing_has_reached(self, published):
        response = _snapshot()

        assert response["success"] is True
        assert response["append_dim"] == "y"
        assert (response["rows_from"], response["rows_written"]) == (0, 5)
        assert response["rows_total"] == ROWS

    def test_asking_from_a_row_returns_only_the_rows_after_it(self, published):
        response = _snapshot(since_rows=3)

        assert (response["rows_from"], response["rows_written"]) == (3, 5)
        assert response["rows_total"] == ROWS
        rows = unpack_snapshot(response["_binary_payload"])["data"]["signal"]
        assert np.asarray(rows).shape == (2, COLS)

    def test_the_rows_returned_are_the_ones_the_client_is_missing(self, published):
        response = _snapshot(since_rows=3)
        rows = np.asarray(
            unpack_snapshot(response["_binary_payload"])["data"]["signal"]
        )

        assert np.array_equal(rows, published["signal"].values[3:5])

    def test_a_caught_up_client_is_sent_no_rows_at_all(self, published):
        response = _snapshot(since_rows=5)

        assert (response["rows_from"], response["rows_written"]) == (5, 5)
        rows = unpack_snapshot(response["_binary_payload"])["data"]["signal"]
        assert np.asarray(rows).shape == (0, COLS)

    def test_a_client_claiming_rows_this_run_does_not_have_gets_everything(
        self, published
    ):
        """Its copy belongs to another run, so its row count means nothing here."""
        response = _snapshot(since_rows=ROWS + 5)

        assert response["rows_from"] == 0
        rows = unpack_snapshot(response["_binary_payload"])["data"]["signal"]
        assert np.asarray(rows).shape == (ROWS, COLS)

    def test_a_dataset_with_no_append_dim_is_sent_whole(self):
        ds = xr.Dataset(
            {"a": (("y",), np.zeros(3)), "b": (("x",), np.zeros(3))},
            coords={"y": np.arange(3), "x": np.arange(3)},
        )
        server.register_snapshot_provider("run-1", lambda: ds)

        response = _snapshot(since_rows=2)

        assert "rows_from" not in response
        assert response["success"] is True

    def test_the_rows_left_out_are_not_on_the_wire(self, published):
        """
        The saving is the omitted rows' bytes -- the JSON header is the same
        either way, and only dominates at this toy size. A whole snapshot
        carries 8 rows, a since_rows=4 one carries the single written row
        after it, so 7 rows of float64 never leave the producer.

        """
        whole = _snapshot()["_binary_payload"]
        partial = _snapshot(since_rows=4)["_binary_payload"]

        assert len(whole) - len(partial) >= 7 * COLS * 8

    @pytest.mark.parametrize("since_rows", [-1, True, 1.5, "2"])
    def test_invalid_row_offsets_are_rejected(self, published, since_rows):
        response = _snapshot(since_rows=since_rows)

        assert response["success"] is False
        assert "since_rows" in response["error"]


class TestIncrementalClient:
    def test_the_client_requests_and_describes_only_the_missing_rows(self, published):
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        delta = client.open_live_measurement_sync("run-1", endpoint, since_rows=3)

        assert delta.sizes == {"y": 2, "x": COLS}
        np.testing.assert_array_equal(
            delta["signal"].values, published["signal"].values[3:5]
        )
        assert delta.encoding["qimchi_connect_rows"] == {
            "append_dim": "y",
            "rows_from": 3,
            "rows_written": 5,
            "rows_total": ROWS,
        }

    def test_a_full_snapshot_advertises_zero_as_its_start(self, published):
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        whole = client.open_live_measurement_sync("run-1", endpoint)

        assert whole.encoding["qimchi_connect_rows"]["rows_from"] == 0

    def test_a_non_rowwise_or_older_response_remains_a_full_snapshot(self):
        dataset = xr.Dataset(
            {"a": (("y",), np.zeros(3)), "b": (("x",), np.ones(3))},
            coords={"y": np.arange(3), "x": np.arange(3)},
        )
        server.register_snapshot_provider("run-1", lambda: dataset)
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        whole = client.open_live_measurement_sync("run-1", endpoint, since_rows=2)

        xr.testing.assert_identical(whole, dataset)
        assert "qimchi_connect_rows" not in whole.encoding

    @pytest.mark.parametrize("since_rows", [-1, True, 1.5, "2"])
    def test_invalid_row_offsets_fail_before_opening_a_connection(self, since_rows):
        with pytest.raises((TypeError, ValueError), match="since_rows"):
            client.open_live_measurement_sync(
                "run-1", "ws://localhost:1", since_rows=since_rows
            )
