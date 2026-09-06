"""
Publish a live xarray dataset to Qimchi from a framework-neutral acquisition
loop -- no QCoDeS, qanary, or Quantify required.

Run with:  python examples/generic_producer.py

This is the pattern to follow for any measurement library qimchi-connect does
not already know about: build/update an `xarray.Dataset` yourself, and hand
`live_measurement` a callback that returns a consistent snapshot of it.

"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import xarray as xr

from qimchi_connect import live_measurement

# Tunables, kept at module scope so the tests can shrink the run.
NUM_POINTS = 50
POINT_DELAY = 0.05


def main() -> Path:
    """
    Run a dummy acquisition loop, published live for its duration.

    Returns:
        Path: The netCDF file the finished measurement was written to.

    """
    voltages = np.linspace(-1.0, 1.0, NUM_POINTS)
    currents = np.full(NUM_POINTS, np.nan)

    # The snapshot callback can run on whatever thread a Qimchi client's
    # request arrives on, while `acquire` below keeps mutating `currents` on
    # this thread -- guard the shared array with a lock so a snapshot never
    # observes a half-written value, and return a copy rather than a view.
    lock = threading.Lock()

    def snapshot() -> xr.Dataset:
        with lock:
            data = currents.copy()
        return xr.Dataset(
            {"current": ("voltage", data)},
            coords={"voltage": voltages},
            attrs={"units_current": "A", "units_voltage": "V"},
        )

    def acquire() -> None:
        for i, voltage in enumerate(voltages):
            time.sleep(POINT_DELAY)  # replace with a real instrument read
            with lock:
                currents[i] = np.sin(voltage * np.pi) + 0.01 * np.random.randn()

    with live_measurement(
        measurement_id="generic-demo-sweep",
        snapshot=snapshot,
        metadata={"source_package": "generic-example"},
    ) as registration:
        acquire()

        # Persist a final copy and point Qimchi at it, so the measurement
        # stays viewable on disk after this live session ends.
        final_path = Path("./generic_demo_sweep.nc").resolve()
        snapshot().to_netcdf(final_path)
        registration.update_disk_path(final_path)

    print(f"Done. Dataset stored at: {final_path}")
    return final_path


if __name__ == "__main__":
    main()
