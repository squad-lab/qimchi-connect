"""
Tests for QCoDeSSnapshotProvider.

The class handles two QCoDeS-specific constraints that a plain
``lambda: dataset.to_xarray_dataset()`` does not: a thread-affine SQLite
connection, and batched writes that make a naive read return stale data. Most
tests use a fake datasaver, so they run without qcodes installed and can
assert on call ordering; the ones at the bottom run against a real QCoDeS
measurement.

"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import xarray as xr

from qimchi_connect import QCoDeSSnapshotProvider, producer


class FakeDatasaver:
    """
    Stands in for a QCoDeS datasaver, recording the order of calls.

    The order matters -- the provider must flush before it reads -- and is
    only observable if the sequence is recorded.

    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.results: list[tuple] = []
        self.points = 0

    def flush_data_to_database(self, block: bool = False) -> None:
        self.calls.append(f"flush(block={block})")

    def add_result(self, *results: object) -> None:
        self.calls.append("add_result")
        self.results.append(results)
        self.points += 1

    @property
    def dataset(self) -> FakeDatasaver:
        return self

    def to_xarray_dataset(self) -> xr.Dataset:
        self.calls.append("to_xarray_dataset")
        # Grows with the run, so a stale snapshot is detectable by its size.
        return xr.Dataset(
            {"v": ("i", np.arange(float(self.points)))},
            coords={"i": np.arange(self.points)},
        )


class TestRefreshSemantics:
    """What refresh() does, and in what order."""

    def test_it_flushes_before_reading(self):
        """
        QCoDeS batches writes, so reading without flushing first can miss the
        points just recorded.

        """
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver)

        provider.refresh(force=True)

        assert saver.calls == ["flush(block=True)", "to_xarray_dataset"]

    def test_the_flush_blocks(self):
        """A non-blocking flush would not guarantee the read sees the points."""
        saver = FakeDatasaver()

        QCoDeSSnapshotProvider(saver).refresh(force=True)

        assert "flush(block=True)" in saver.calls

    def test_the_snapshot_reflects_points_recorded_so_far(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver)
        for _ in range(3):
            saver.add_result()

        provider.refresh(force=True)

        assert provider()["v"].size == 3


class TestThrottling:
    """
    to_xarray_dataset re-reads the whole run, so refreshing per point is
    quadratic. refresh_interval bounds how often that happens.

    """

    def test_repeated_refreshes_inside_the_interval_do_no_work(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=3600)

        provider.refresh()
        reads_after_first = saver.calls.count("to_xarray_dataset")
        for _ in range(50):
            provider.refresh()

        assert reads_after_first == 1
        assert saver.calls.count("to_xarray_dataset") == 1

    def test_force_bypasses_the_throttle(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=3600)

        provider.refresh()
        provider.refresh(force=True)

        assert saver.calls.count("to_xarray_dataset") == 2

    def test_a_zero_interval_refreshes_every_time(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=0)

        for _ in range(5):
            provider.refresh()

        assert saver.calls.count("to_xarray_dataset") == 5

    def test_a_negative_interval_is_rejected(self):
        with pytest.raises(ValueError, match="refresh_interval"):
            QCoDeSSnapshotProvider(FakeDatasaver(), refresh_interval=-1)


class TestSnapshotCallbackIsThreadSafe:
    """__call__ runs on the server's thread and must never touch the datasaver."""

    def test_calling_the_provider_does_not_touch_the_datasaver(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver)
        provider.refresh(force=True)
        saver.calls.clear()

        for _ in range(10):
            provider()

        assert saver.calls == []

    def test_it_returns_an_empty_dataset_before_the_first_refresh(self):
        """A client connecting early gets an empty dataset, never an error."""
        assert list(QCoDeSSnapshotProvider(FakeDatasaver())().data_vars) == []

    def test_concurrent_readers_and_a_refreshing_writer_never_tear(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=0)
        stop = threading.Event()
        seen: list[int] = []
        errors: list[BaseException] = []

        def read() -> None:
            try:
                while not stop.is_set():
                    seen.append(provider()["v"].size if provider().data_vars else 0)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        reader = threading.Thread(target=read)
        reader.start()
        try:
            for _ in range(200):
                saver.add_result()
                provider.refresh()
        finally:
            stop.set()
            reader.join()

        assert errors == []
        # Sizes only ever grow: a torn read would show a size going backwards.
        assert seen == sorted(seen)


class TestAddResult:
    """The one-call-per-point convenience."""

    def test_it_records_the_point_and_refreshes(self):
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=0)

        provider.add_result(("p", 1.0), ("q", 2.0))

        assert saver.results == [(("p", 1.0), ("q", 2.0))]
        assert saver.calls == [
            "add_result",
            "flush(block=True)",
            "to_xarray_dataset",
        ]

    def test_the_refresh_it_triggers_is_throttled(self):
        """Per-point calls must not each pay for a full re-read."""
        saver = FakeDatasaver()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=3600)

        for _ in range(20):
            provider.add_result()

        assert saver.calls.count("add_result") == 20
        assert saver.calls.count("to_xarray_dataset") == 1


class TestProducerIntegration:
    """register_live_measurement should treat a self-describing provider as one."""

    def test_it_seeds_the_cache_before_publishing(self):
        """
        A client connecting immediately must see the run so far, so the
        provider is refreshed before the server can be queried.

        """
        saver = FakeDatasaver()
        for _ in range(4):
            saver.add_result()
        provider = QCoDeSSnapshotProvider(saver, refresh_interval=3600)

        with producer.live_measurement("seeded", provider) as registration:
            assert registration.ws_port > 0
            assert provider()["v"].size == 4

    def test_it_takes_source_package_from_the_provider(self):
        provider = QCoDeSSnapshotProvider(FakeDatasaver())

        with producer.live_measurement("described", provider):
            from qimchi_connect import server

            entry = server._provider_entry("described")
            assert entry.metadata["source_package"] == "qcodes"

    def test_an_explicit_metadata_argument_still_wins(self):
        provider = QCoDeSSnapshotProvider(FakeDatasaver())

        with producer.live_measurement(
            "overridden", provider, metadata={"source_package": "mine"}
        ):
            from qimchi_connect import server

            entry = server._provider_entry("overridden")
            assert entry.metadata["source_package"] == "mine"

    def test_a_bare_callable_is_accepted(self):
        """The provider is opt-in: a bare callable is equally valid."""
        dataset = xr.Dataset({"v": ("i", [1.0, 2.0])}, coords={"i": [0, 1]})

        with producer.live_measurement("plain", lambda: dataset) as registration:
            assert registration.ws_port > 0


class TestAgainstRealQCoDeS:
    """
    The thread-affinity constraint, against a real QCoDeS run.

    Skipped when qcodes is not installed; it is an optional extra.

    """

    @staticmethod
    def _run_measurement(tmp_path):
        """Start a real QCoDeS run and return (datasaver context, instruments)."""
        from qcodes.dataset import (
            Measurement,
            initialise_or_create_database_at,
            load_or_create_experiment,
        )
        from qcodes.instrument import Instrument
        from qcodes.instrument_drivers.mock_instruments import (
            DummyInstrument,
            DummyInstrumentWithMeasurement,
        )

        initialise_or_create_database_at(str(tmp_path / "test.db"))
        experiment = load_or_create_experiment(
            experiment_name="provider_test", sample_name="s"
        )
        Instrument.close_all()
        dac = DummyInstrument("dac", gates=["ch1"])
        dmm = DummyInstrumentWithMeasurement("dmm", setter_instr=dac)
        measurement = Measurement(exp=experiment, name="run")
        measurement.register_parameter(dac.ch1)
        measurement.register_parameter(dmm.v1, setpoints=(dac.ch1,))
        return measurement, dac, dmm, Instrument

    def test_a_read_from_another_thread_is_refused(self, tmp_path):
        """
        QCoDeS refuses a cross-thread read of a running run, which is the
        constraint the provider works around.

        """
        pytest.importorskip("qcodes")
        measurement, dac, dmm, Instrument = self._run_measurement(tmp_path)
        failure: list[BaseException] = []

        try:
            with measurement.run() as datasaver:
                dac.ch1.set(0.1)
                datasaver.add_result((dac.ch1, 0.1), (dmm.v1, dmm.v1.get()))

                def read_from_elsewhere() -> None:
                    try:
                        datasaver.dataset.to_xarray_dataset()
                    except BaseException as exc:  # noqa: BLE001 - asserted below
                        failure.append(exc)

                thread = threading.Thread(target=read_from_elsewhere)
                thread.start()
                thread.join()
        finally:
            Instrument.close_all()

        assert failure, "expected QCoDeS' thread-affine connection to refuse"
        assert "thread" in str(failure[0]).lower()

    def test_the_provider_serves_the_same_run_from_another_thread(self, tmp_path):
        pytest.importorskip("qcodes")
        measurement, dac, dmm, Instrument = self._run_measurement(tmp_path)
        result: dict[str, xr.Dataset] = {}
        errors: list[BaseException] = []

        try:
            with measurement.run() as datasaver:
                provider = QCoDeSSnapshotProvider(datasaver, refresh_interval=0)
                for voltage in (0.1, 0.2, 0.3):
                    dac.ch1.set(voltage)
                    provider.add_result((dac.ch1, voltage), (dmm.v1, dmm.v1.get()))

                def read_from_elsewhere() -> None:
                    try:
                        result["dataset"] = provider()
                    except BaseException as exc:  # noqa: BLE001 - asserted below
                        errors.append(exc)

                thread = threading.Thread(target=read_from_elsewhere)
                thread.start()
                thread.join()
        finally:
            Instrument.close_all()

        assert errors == []
        assert result["dataset"]["dmm_v1"].size == 3
