"""
Tests for publishing and closing a measurement by identifier.

A producer whose publish and close happen in different places -- qanary
registers in ``run()`` and closes in a ``finally`` on another method -- can
address an open publication by its identifier instead of carrying the handle
between them or keeping its own mapping.

"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import (
    QanarySnapshotProvider,
    close_live_measurement,
    get_live_registration,
    register_live_measurement,
    registry,
    server,
    update_live_disk_path,
)


@pytest.fixture
def store():
    zarr = pytest.importorskip("zarr")
    memory_store = zarr.storage.MemoryStore()
    xr.Dataset({"signal": ("x", np.arange(3.0))}, coords={"x": np.arange(3)}).to_zarr(
        store=memory_store, consolidated=False
    )
    return memory_store


class TestQanarySnapshotProvider:
    def test_it_reads_the_current_contents_of_the_store(self, store):
        assert QanarySnapshotProvider(store)()["signal"].values.tolist() == [
            0.0,
            1.0,
            2.0,
        ]

    def test_it_sees_writes_made_after_construction(self, store):
        """The sweep keeps writing; every call re-reads."""
        provider = QanarySnapshotProvider(store)
        first = provider()["signal"].size

        xr.Dataset(
            {"signal": ("x", np.arange(6.0))}, coords={"x": np.arange(6)}
        ).to_zarr(store=store, mode="w", consolidated=False)

        assert first == 3
        assert provider()["signal"].size == 6

    def test_it_describes_itself_as_qanary_zarr(self, store):
        with register_live_measurement("m", QanarySnapshotProvider(store)) as handle:
            entry = server._provider_entry(handle.measurement_id)
            assert entry.metadata == {
                "source_package": "qanary",
                "source_format": "zarr",
            }

    def test_an_explicit_metadata_argument_still_wins(self, store):
        with register_live_measurement(
            "m", QanarySnapshotProvider(store), metadata={"source_package": "mine"}
        ):
            assert server._provider_entry("m").metadata["source_package"] == "mine"


class TestIdAddressedOperations:
    def test_an_open_publication_is_resolvable_by_id(self, store):
        registration = register_live_measurement("m", QanarySnapshotProvider(store))

        assert get_live_registration("m") is registration

    def test_an_unpublished_measurement_resolves_to_nothing(self):
        assert get_live_registration("never-published") is None

    def test_closing_by_id_stops_serving_and_ends_the_record(self, store):
        register_live_measurement("m", QanarySnapshotProvider(store))

        assert close_live_measurement("m") is True

        assert server._provider_entry("m") is None
        assert registry.get_measurement("m").live_status is False
        assert get_live_registration("m") is None

    def test_closing_an_unpublished_measurement_reports_that(self):
        assert close_live_measurement("never-published") is False

    def test_closing_twice_is_harmless(self, store):
        register_live_measurement("m", QanarySnapshotProvider(store))

        assert close_live_measurement("m") is True
        assert close_live_measurement("m") is False

    def test_the_disk_path_can_be_updated_by_id(self, store, tmp_path):
        register_live_measurement("m", QanarySnapshotProvider(store))
        final = tmp_path / "m.nc"

        assert update_live_disk_path("m", final) is True

        assert registry.get_measurement("m").fpath == str(final)
        assert get_live_registration("m").disk_path == str(final)

    def test_updating_an_unpublished_measurement_reports_that(self, tmp_path):
        assert update_live_disk_path("never-published", tmp_path / "x.nc") is False


class TestMaintenanceLogging:
    def test_findings_are_reported_by_the_library(self, store, caplog, monkeypatch):
        """
        A producer should not have to import the registry just to log what the
        sweep found.

        """
        monkeypatch.setattr(
            registry,
            "maintain_registry",
            lambda **_: registry.RegistryMaintenanceResult(("stale-1",), 2),
        )

        with caplog.at_level("INFO", logger="qimchi_connect.producer"):
            register_live_measurement("m", QanarySnapshotProvider(store))

        assert "marked 1 stale and deleted 2" in caplog.text

    def test_a_failing_sweep_does_not_block_publication(self, store, caplog):
        def explode(**_):
            raise RuntimeError("locked")

        with caplog.at_level("WARNING", logger="qimchi_connect.producer"):
            import qimchi_connect.producer as producer_module

            original = producer_module.registry.maintain_registry
            producer_module.registry.maintain_registry = explode
            try:
                register_live_measurement("m", QanarySnapshotProvider(store))
            finally:
                producer_module.registry.maintain_registry = original

        assert "Could not maintain the live registry" in caplog.text
        assert server._provider_entry("m") is not None
