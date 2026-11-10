"""Publish a simulated QCoDeS measurement while it runs.

Run with:  python examples/qcodes_measurement.py
Requires:  pip install qimchi-connect[examples]   (adds qcodes)

The example follows the standard QCoDeS pattern from
https://microsoft.github.io/Qcodes/examples/DataSet/Performing-measurements-using-qcodes-parameters-and-dataset.html
and adds ``QCoDeSSnapshotProvider`` plus ``live_measurement``. Qimchi reads the
SQLite database after live publication ends.

QCoDeS binds its SQLite connection to the measurement thread and buffers
writes. The provider handles both constraints; use its ``add_result`` method
instead of calling ``datasaver.add_result`` directly.
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

# Module-level settings let the tests shorten the run.
VOLTAGES = [i / 10 for i in range(-50, 51)]
POINT_DELAY = 0.05


def main(db_path: Path | None = None) -> Path:
    """
    Run and publish a simulated QCoDeS sweep.

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
            # The provider reads QCoDeS data on the measurement thread.
            live = QCoDeSSnapshotProvider(datasaver)

            # Registration creates the initial cached snapshot.
            with live_measurement(
                measurement_id=f"qcodes-{datasaver.dataset.captured_run_id}",
                snapshot=live,
                disk_path=db_path,
            ):
                for voltage in VOLTAGES:
                    dac.ch1.set(voltage)
                    reading = dmm.v1.get()
                    # Record the point and refresh the cache when due.
                    live.add_result((dac.ch1, voltage), (dmm.v1, reading))
                    time.sleep(POINT_DELAY)  # replace with real timing

                # Ensure the final point is included before publication closes.
                live.refresh(force=True)
    finally:
        Instrument.close_all()

    print(f"Done. Dataset stored at: {db_path}")
    return db_path


if __name__ == "__main__":
    main()
