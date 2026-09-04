"""
Tests for QuantifySnapshotProvider.

Quantify flushes the in-progress dataset to disk as it acquires and
``load_dataset`` re-reads it, so unlike QCoDeS there is no thread affinity to
work around. What the class owns is the convention -- id, disk path, producer
metadata -- and the window at the start of a run where the file is not yet
readable.

These tests use a fake ``quantify_core`` module because it is an optional
dependency. The fake module also records how the provider calls Quantify.

"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import QuantifySnapshotProvider, live_measurement, server

TUID = "20260904-101500-123-abcdef"


class FakeQuantify:
    """Stands in for quantify_core.data.handling."""

    def __init__(self, container=None, dataset=None, error=None):
        self.container = container
        self.dataset = dataset
        self.error = error
        self.loads = 0

    def load_dataset(self, tuid):
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.dataset

    def locate_experiment_container(self, tuid):
        if self.container is None:
            raise FileNotFoundError(tuid)
        return str(self.container)


@pytest.fixture
def quantify(monkeypatch):
    """Install a fake quantify_core for the duration of a test."""
    fake = FakeQuantify(dataset=xr.Dataset({"z": ("dim_0", np.arange(4.0))}))

    handling = types.ModuleType("quantify_core.data.handling")
    handling.load_dataset = fake.load_dataset
    handling.locate_experiment_container = fake.locate_experiment_container
    package = types.ModuleType("quantify_core")
    data = types.ModuleType("quantify_core.data")

    monkeypatch.setitem(sys.modules, "quantify_core", package)
    monkeypatch.setitem(sys.modules, "quantify_core.data", data)
    monkeypatch.setitem(sys.modules, "quantify_core.data.handling", handling)
    return fake


class TestConventions:
    def test_the_measurement_id_is_the_prefixed_tuid(self, quantify):
        assert QuantifySnapshotProvider(TUID).measurement_id == f"quantify-{TUID}"

    def test_the_tuid_is_available_unchanged(self, quantify):
        assert QuantifySnapshotProvider(TUID).tuid == TUID

    def test_it_names_quantify_as_the_source(self, quantify):
        assert QuantifySnapshotProvider(TUID).metadata == {
            "source_package": "quantify",
            "source_format": "hdf5",
        }

    def test_the_disk_path_is_the_datasets_own_file(self, quantify, tmp_path):
        quantify.container = tmp_path / "experiment"

        assert QuantifySnapshotProvider(TUID).disk_path == (
            tmp_path / "experiment" / "dataset.hdf5"
        )

    def test_the_disk_path_is_absent_before_the_container_exists(self, quantify):
        """Publishing can start before Quantify has created the directory."""
        quantify.container = None

        assert QuantifySnapshotProvider(TUID).disk_path is None


class TestReading:
    def test_it_returns_the_current_dataset(self, quantify):
        assert QuantifySnapshotProvider(TUID)()["z"].size == 4

    def test_it_re_reads_on_every_call(self, quantify):
        """The run is still acquiring; a cached first read would freeze it."""
        provider = QuantifySnapshotProvider(TUID)

        provider()
        quantify.dataset = xr.Dataset({"z": ("dim_0", np.arange(9.0))})

        assert provider()["z"].size == 9
        assert quantify.loads == 2

    def test_an_unreadable_dataset_yields_an_empty_dataset_not_an_error(self, quantify):
        """
        Quantify rewrites the file as it acquires, so a client polling during a
        write must not get an exception instead of the run.

        """
        quantify.error = OSError("file is being written")

        assert list(QuantifySnapshotProvider(TUID)().data_vars) == []

    def test_a_failed_read_serves_the_previous_snapshot(self, quantify):
        provider = QuantifySnapshotProvider(TUID)
        provider()

        quantify.error = OSError("file is being written")

        assert provider()["z"].size == 4


class TestPublication:
    def test_it_publishes_under_its_own_id_and_metadata(self, quantify, tmp_path):
        quantify.container = tmp_path / "experiment"
        provider = QuantifySnapshotProvider(TUID)

        with live_measurement(
            provider.measurement_id, provider, disk_path=provider.disk_path
        ) as registration:
            entry = server._provider_entry(provider.measurement_id)

            assert registration.measurement_id == f"quantify-{TUID}"
            assert entry.metadata["source_package"] == "quantify"
            assert registration.disk_path.endswith("dataset.hdf5")


def test_without_quantify_core_it_says_which_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "quantify_core", None)

    with pytest.raises(ImportError, match=r"qimchi-connect\[quantify\]"):
        QuantifySnapshotProvider(TUID)
