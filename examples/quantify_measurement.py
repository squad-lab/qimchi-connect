"""
Stream a Quantify measurement to Qimchi while it runs.

Run with:  python examples/quantify_measurement.py
Requires:  pip install "qimchi-connect[quantify]"

The example uses Quantify's MeasurementControl with software-only QCoDeS
parameters, so it needs no laboratory hardware. While the sweep is running,
Qimchi's Explorer shows ``quantify-<tuid>`` as a live measurement. When the
run finishes, Qimchi can fall back to Quantify's ``dataset.hdf5`` file.

MeasurementControl creates the TUID inside ``run()``. A small companion
thread therefore watches Quantify's public data-discovery API for the new run
and publishes it as soon as its experiment directory appears. The snapshot
provider reopens the HDF5 dataset on each request, which is safe from the
WebSocket server thread and resilient to reads that overlap a Quantify write.

"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from qcodes import validators
from qcodes.parameters import ManualParameter
from quantify_core.data.handling import (
    get_tuids_containing,
    locate_experiment_container,
    set_datadir,
)
from quantify_core.measurement import MeasurementControl

from qimchi_connect import QuantifySnapshotProvider, live_measurement

# Tunables, kept at module scope so tests can shrink the run.
EXPERIMENT_NAME = "qimchi_connect_quantify_demo"
X_VALUES = np.linspace(-1.0, 1.0, 21)
Y_VALUES = np.linspace(0.0, 2.0, 21)
POINT_DELAY = 0.05
DISCOVERY_TIMEOUT = 30.0


class DemoSignal:
    """Software-only Quantify gettable used by the example."""

    name = "signal"
    label = "Signal"
    unit = "A"

    def __init__(self, x: ManualParameter, y: ManualParameter) -> None:
        """
        Store the parameters that determine the simulated signal.

        Args:
            x (ManualParameter): First settable.
            y (ManualParameter): Second settable.

        """
        self._x = x
        self._y = y

    def get(self) -> float:
        """
        Return one simulated measurement value.

        Returns:
            float: Signal at the current setpoints.

        """
        time.sleep(POINT_DELAY)
        return float(np.sin(self._x()) + np.cos(self._y()))


def _known_tuids() -> set[str]:
    """
    Return existing runs with the example's name.

    Returns:
        set[str]: TUID strings already present in the data directory.

    """
    try:
        return {str(tuid) for tuid in get_tuids_containing(EXPERIMENT_NAME)}
    except FileNotFoundError:
        return set()


def main(data_directory: Path | None = None) -> Path:
    """
    Run a dummy two-dimensional Quantify sweep and publish it live.

    Args:
        data_directory (Path | None): Quantify data directory. Defaults to
            ``./qimchi_connect_quantify_data``.

    Returns:
        Path: The completed Quantify ``dataset.hdf5`` file.

    Raises:
        RuntimeError: If the new TUID cannot be discovered or publication
            fails.

    """
    data_directory = (
        data_directory or Path("./qimchi_connect_quantify_data")
    ).resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    set_datadir(data_directory)

    existing_tuids = _known_tuids()
    measurement_done = threading.Event()
    published = threading.Event()
    publication_errors: list[Exception] = []

    def publish_new_run() -> None:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT
        try:
            while time.monotonic() < deadline:
                new_tuids = _known_tuids() - existing_tuids
                if new_tuids:
                    # TUIDs begin with a sortable timestamp.
                    tuid = max(new_tuids)
                    provider = QuantifySnapshotProvider(tuid)
                    with live_measurement(
                        provider.measurement_id,
                        provider,
                        disk_path=provider.disk_path,
                    ):
                        published.set()
                        measurement_done.wait()
                    return
                if measurement_done.is_set():
                    break
                time.sleep(0.05)
            raise RuntimeError("the Quantify run's TUID was not discovered")
        except Exception as exc:
            publication_errors.append(exc)

    publisher = threading.Thread(
        target=publish_new_run,
        daemon=True,
        name="qimchi-connect-quantify-publisher",
    )
    publisher.start()

    measurement_control = MeasurementControl("qimchi_connect_demo_control")
    x = ManualParameter(
        "x", label="X voltage", unit="V", vals=validators.Numbers(), initial_value=0
    )
    y = ManualParameter(
        "y", label="Y voltage", unit="V", vals=validators.Numbers(), initial_value=0
    )

    try:
        measurement_control.update_interval(0.15)
        measurement_control.settables([x, y])
        measurement_control.setpoints_grid([X_VALUES, Y_VALUES])
        measurement_control.gettables(DemoSignal(x, y))
        dataset = measurement_control.run(EXPERIMENT_NAME)
    finally:
        measurement_done.set()
        publisher.join(timeout=DISCOVERY_TIMEOUT + 1.0)
        measurement_control.close()

    if publisher.is_alive():
        raise RuntimeError("the Quantify publisher thread did not stop")
    if publication_errors:
        raise RuntimeError(
            "failed to publish the Quantify run"
        ) from publication_errors[0]
    if not published.is_set():
        raise RuntimeError("the Quantify run finished without being published")

    tuid = str(dataset.attrs["tuid"])
    dataset_path = Path(locate_experiment_container(tuid)) / "dataset.hdf5"
    print(f"Done. Dataset stored at: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    main()
