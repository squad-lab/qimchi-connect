"""
Examples should run without error.

All are hardware-free -- the QCoDeS one drives QCoDeS' own
mock instruments and the Quantify one uses software parameters.

Their loops are shrunk through the module-level tunables rather than by
running the full sweep, so the suite stays fast while still taking the code
path the example takes.

"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(name: str):
    """
    Import an example script by path, without needing it on sys.path.

    Args:
        name (str): File name of the example, including the suffix.

    Returns:
        Any: The imported module.

    """
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_example_is_covered_by_a_test():
    """
    Every script in examples/ has a test that runs it. A new example is added
    to this list together with that test.

    """
    tested = {
        "generic_producer.py",
        "qcodes_measurement.py",
        "quantify_measurement.py",
    }

    assert {path.name for path in EXAMPLES.glob("*.py")} == tested


def test_generic_producer_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = load_example("generic_producer.py")
    monkeypatch.setattr(module, "NUM_POINTS", 5, raising=True)
    monkeypatch.setattr(module, "POINT_DELAY", 0.0, raising=True)

    module.main()

    written = tmp_path / "generic_demo_sweep.nc"
    assert written.exists(), "the example should leave its measurement on disk"


def test_qcodes_measurement_runs(tmp_path, monkeypatch):
    pytest.importorskip("qcodes", reason="qcodes is an optional 'examples' extra")

    monkeypatch.chdir(tmp_path)
    module = load_example("qcodes_measurement.py")
    monkeypatch.setattr(module, "VOLTAGES", [0.0, 0.1, 0.2], raising=True)
    monkeypatch.setattr(module, "POINT_DELAY", 0.0, raising=True)

    db_path = module.main(db_path=tmp_path / "demo.db")

    assert db_path.exists(), "the example should leave its QCoDeS database on disk"


def test_the_qcodes_example_publishes_what_it_measured(tmp_path, monkeypatch):
    """
    The example's purpose is that the live snapshot tracks the run, so check
    the published data arrived rather than only that the script completed.

    """
    pytest.importorskip("qcodes", reason="qcodes is an optional 'examples' extra")

    from qimchi_connect import producer

    monkeypatch.chdir(tmp_path)
    module = load_example("qcodes_measurement.py")
    monkeypatch.setattr(module, "VOLTAGES", [0.0, 0.1, 0.2, 0.3], raising=True)
    monkeypatch.setattr(module, "POINT_DELAY", 0.0, raising=True)

    published: list = []
    real_live_dataset = producer.live_measurement

    def capture(measurement_id, snapshot, **options):
        published.append(snapshot)
        return real_live_dataset(measurement_id, snapshot, **options)

    monkeypatch.setattr(module, "live_measurement", capture, raising=True)
    module.main(db_path=tmp_path / "demo.db")

    assert len(published) == 1
    snapshot = published[0]()
    assert snapshot["dmm_v1"].size == 4, "every point should reach the snapshot"


def _install_fake_quantify(monkeypatch, tmp_path):
    """Install the small Quantify surface used by the runnable example."""
    state = {
        "tuids": [],
        "dataset": None,
        "container": tmp_path / "quantify-run",
    }

    def set_datadir(_path):
        state["container"].mkdir(parents=True, exist_ok=True)

    def get_tuids_containing(_name):
        return list(state["tuids"])

    def locate_experiment_container(_tuid):
        return state["container"]

    def load_dataset(_tuid):
        return state["dataset"].copy(deep=True)

    class FakeMeasurementControl:
        """Exercise the example without installing Quantify's retired stack."""

        def __init__(self, _name):
            self.setpoints = None

        def update_interval(self, _interval):
            pass

        def settables(self, _parameters):
            pass

        def setpoints_grid(self, values):
            self.setpoints = values

        def gettables(self, _gettable):
            pass

        def run(self, name):
            tuid = "20260904-120000-000-abcdef"
            x_values, y_values = self.setpoints
            state["dataset"] = xr.Dataset(
                {
                    "signal": (
                        ("x", "y"),
                        np.add.outer(x_values, y_values),
                    )
                },
                coords={"x": x_values, "y": y_values},
                attrs={"tuid": tuid, "name": name},
            )
            (state["container"] / "dataset.hdf5").write_bytes(b"fake hdf5")
            state["tuids"].append(tuid)
            return state["dataset"]

        def close(self):
            pass

    handling = types.ModuleType("quantify_core.data.handling")
    handling.get_tuids_containing = get_tuids_containing
    handling.load_dataset = load_dataset
    handling.locate_experiment_container = locate_experiment_container
    handling.set_datadir = set_datadir

    measurement = types.ModuleType("quantify_core.measurement")
    measurement.MeasurementControl = FakeMeasurementControl

    package = types.ModuleType("quantify_core")
    data = types.ModuleType("quantify_core.data")
    monkeypatch.setitem(sys.modules, "quantify_core", package)
    monkeypatch.setitem(sys.modules, "quantify_core.data", data)
    monkeypatch.setitem(sys.modules, "quantify_core.data.handling", handling)
    monkeypatch.setitem(sys.modules, "quantify_core.measurement", measurement)

    return state


def test_quantify_measurement_runs_and_publishes(tmp_path, monkeypatch):
    """The Quantify example should publish its run and return its HDF5 path."""
    state = _install_fake_quantify(monkeypatch, tmp_path)
    module = load_example("quantify_measurement.py")
    monkeypatch.setattr(module, "X_VALUES", np.linspace(0.0, 1.0, 3), raising=True)
    monkeypatch.setattr(module, "Y_VALUES", np.linspace(0.0, 1.0, 2), raising=True)
    monkeypatch.setattr(module, "POINT_DELAY", 0.0, raising=True)

    published = []
    real_live_measurement = module.live_measurement

    def capture(measurement_id, snapshot, **options):
        published.append((measurement_id, snapshot, options))
        return real_live_measurement(measurement_id, snapshot, **options)

    monkeypatch.setattr(module, "live_measurement", capture, raising=True)
    dataset_path = module.main(data_directory=tmp_path / "quantify-data")

    assert dataset_path.exists()
    assert len(published) == 1
    measurement_id, provider, options = published[0]
    assert measurement_id == f"quantify-{state['tuids'][0]}"
    assert options["disk_path"] == dataset_path
    assert provider()["signal"].shape == (3, 2)
