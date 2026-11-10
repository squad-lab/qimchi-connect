"""Publish an xarray dataset without a framework-specific provider.

Run with:  python examples/generic_producer.py

The callback pattern shown here works with any acquisition library: update the
data under a lock and return an independent ``xarray.Dataset`` snapshot.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import xarray as xr

from qimchi_connect import live_measurement

# Module-level settings let the tests shorten the run.
NUM_POINTS = 50
POINT_DELAY = 0.05


def main() -> Path:
    """
    Run and publish a simulated acquisition.

    Returns:
        Path: The netCDF file the finished measurement was written to.

    """
    voltages = np.linspace(-1.0, 1.0, NUM_POINTS)
    currents = np.full(NUM_POINTS, np.nan)

    # The server reads snapshots from another thread. Lock the shared array and
    # return a copy so each response is consistent.
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

        # Save the final dataset and advertise its path for later viewing.
        final_path = Path("./generic_demo_sweep.nc").resolve()
        snapshot().to_netcdf(final_path)
        registration.update_disk_path(final_path)

    print(f"Done. Dataset stored at: {final_path}")
    return final_path


if __name__ == "__main__":
    main()
