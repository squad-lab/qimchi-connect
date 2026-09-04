"""
Stream a QCoDeS measurement to Qimchi while it runs.

Run with:  python examples/qcodes_measurement.py
Requires:  pip install qimchi-connect[examples]   (adds qcodes)

While this script is running, Qimchi's Explorer shows a live node for the
measurement id below that updates as points are added -- no matter where the
QCoDeS sqlite database itself lives. Once the `with meas.run() as datasaver`
block exits, Qimchi falls back to reading the finished measurement from that
sqlite file directly (Qimchi's loader already understands native QCoDeS
`.db` files).

This mirrors the standard QCoDeS pattern from
https://microsoft.github.io/Qcodes/examples/DataSet/Performing-measurements-using-qcodes-parameters-and-dataset.html
-- the only addition is wrapping `datasaver` in a `QCoDeSSnapshotProvider`
and handing that to `live_measurement`.

A generic producer would pass
`snapshot=lambda: datasaver.dataset.to_xarray_dataset()`. That does not work
for QCoDeS: the run's sqlite connection is thread-affine and its writes are
batched, so the callback raises `sqlite3.ProgrammingError` on the server
thread and can return stale data otherwise. `QCoDeSSnapshotProvider` handles
both -- see its docstring. Use its `add_result` in place of
`datasaver.add_result` and the live snapshot keeps itself current.

"""

from __future__ import annotations

import time
from pathlib import Path

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

from qimchi_connect import QCoDeSSnapshotProvider, live_measurement

# Tunables, kept at module scope so the tests can shrink the run.
VOLTAGES = [i / 10 for i in range(-50, 51)]
POINT_DELAY = 0.05


def main(db_path: Path | None = None) -> Path:
    """
    Run a dummy QCoDeS sweep, published live for the duration.

    Args:
        db_path (Path | None): Where to put the QCoDeS sqlite database.
            Defaults to the current working directory.

    Returns:
        Path: The database the measurement was written to.

    """
    db_path = (db_path or Path("./qimchi_connect_qcodes_demo.db")).resolve()
    initialise_or_create_database_at(str(db_path))
    experiment = load_or_create_experiment(
        experiment_name="qimchi_connect_demo", sample_name="dummy_sample"
    )

    Instrument.close_all()
    dac = DummyInstrument("dac", gates=["ch1"])
    dmm = DummyInstrumentWithMeasurement("dmm", setter_instr=dac)

    meas = Measurement(exp=experiment, name="qimchi_connect_demo_sweep")
    meas.register_parameter(dac.ch1)
    meas.register_parameter(dmm.v1, setpoints=(dac.ch1,))

    try:
        with meas.run() as datasaver:
            # Wrap the datasaver in a live snapshot provider for Qimchi
            live = QCoDeSSnapshotProvider(datasaver)

            # live_measurement seeds the provider's cache before publishing
            with live_measurement(
                measurement_id=f"qcodes-{datasaver.dataset.captured_run_id}",
                snapshot=live,
                disk_path=db_path,
            ):
                for voltage in VOLTAGES:
                    dac.ch1.set(voltage)
                    reading = dmm.v1.get()
                    # Records the point and refreshes the live snapshot, which
                    # is throttled internally -- see QCoDeSSnapshotProvider.
                    live.add_result((dac.ch1, voltage), (dmm.v1, reading))
                    time.sleep(POINT_DELAY)  # replace with real timing

                # The loop's last refresh may have been throttled away, so make
                # the published snapshot final before the registration closes.
                live.refresh(force=True)
    finally:
        Instrument.close_all()

    print(f"Done. Dataset stored at: {db_path}")
    return db_path


if __name__ == "__main__":
    main()
