# Qimchi Connect

[![pipeline](https://gitlab.com/squad-lab/qimchi-connect/badges/main/pipeline.svg?ignore_skipped=true&key_text=pipeline&key_width=60)](https://gitlab.com/squad-lab/qimchi-connect/-/pipelines?ref=main)
[![tests](https://gitlab.com/squad-lab/qimchi-connect/badges/main/pipeline.svg?job=pytest%3A%20%5B3.13%5D&ignore_skipped=true&key_text=tests&key_width=40)](https://gitlab.com/squad-lab/qimchi-connect/-/pipelines?ref=main)
[![coverage](https://gitlab.com/squad-lab/qimchi-connect/badges/main/coverage.svg?key_text=coverage&key_width=64)](https://gitlab.com/squad-lab/qimchi-connect/-/jobs)
[![latest release](https://gitlab.com/squad-lab/qimchi-connect/-/badges/release.svg?key_text=release&key_width=54)](https://gitlab.com/squad-lab/qimchi-connect/-/releases)
[![PyPI version](https://img.shields.io/pypi/v/qimchi-connect)](https://pypi.org/project/qimchi-connect/)

`qimchi-connect` publishes real-time `xarray.Dataset` snapshots to Qimchi without
requiring a specific measurement framework or storage format.

To publish a measurement, provide a callback that returns the current
`xarray.Dataset`. Built-in providers support QCoDeS, qcutils, and Quantify.

---

- [Qimchi Connect](#qimchi-connect)
  - [Installation](#installation)
  - [Publishing a measurement](#publishing-a-measurement)
  - [Supported frameworks](#supported-frameworks)
  - [Runnable examples](#runnable-examples)
  - [Repairing discovery records](#repairing-discovery-records)
  - [Runtime behavior](#runtime-behavior)
  - [Development](#development)
  - [License](#license)

---

## Installation

Install the latest release from PyPI:

```console
pip install qimchi-connect
```

For a project managed with `uv`:

```console
uv add qimchi-connect
```

To install the latest development version from the `preview` branch:

```console
pip install "qimchi-connect @ git+https://gitlab.com/squad-lab/qimchi-connect.git@preview"
```

For a project managed with `uv`:

```console
uv add "qimchi-connect @ git+https://gitlab.com/squad-lab/qimchi-connect.git@preview"
```

Qimchi Connect requires Python 3.13 or later. Python 3.13 is recommended, and
CI also tests Python 3.14.

Optional extras are listed below. For example, install the Quantify extra with
`pip install "qimchi-connect[quantify]"`.

| Extra | Adds | For |
|---|---|---|
| `examples` | `qcodes` | `QCoDeSSnapshotProvider` and `examples/qcodes_measurement.py` |
| `quantify` | `quantify-core` | `QuantifySnapshotProvider` and `examples/quantify_measurement.py` |
| `dev` | pre-commit, ruff, pytest | Development |
| `test` | pytest, pytest-cov | Tests |

## Publishing a measurement

The snapshot callback is evaluated when a client requests data. It should
return a consistent view of the measurement and use the producer's own lock
when the underlying data can change concurrently.

```python
from qimchi_connect import live_measurement

with live_measurement(
    measurement_id="experiment-42",
    snapshot=lambda: current_dataset.copy(deep=True),
    disk_path="data/experiment-42.nc",
):
    perform_measurement()
```

## Supported frameworks

Pass a framework provider to `live_measurement` instead of a callback. A
provider can prepare the first snapshot, set `source_package`, and derive the
measurement ID and disk path.

> [!tip]
> The WebSocket server runs on a background thread. A framework handle may not be safe to read from that thread. For example, QCoDeS uses a thread-bound SQLite connection and batches writes. Direct calls to `datasaver.dataset.to_xarray_dataset()` from the server thread can raise `sqlite3.ProgrammingError`, and reads may not include the latest result.

`QCoDeSSnapshotProvider` manages the cache, locking, write flushing, and refresh
rate:

```python
# QCoDeS
from qimchi_connect import QCoDeSSnapshotProvider, live_measurement

with meas.run() as datasaver:
    live = QCoDeSSnapshotProvider(datasaver)
    with live_measurement(f"qcodes-{datasaver.dataset.captured_run_id}", live):
        for voltage in voltages:
            dac.ch1.set(voltage)
            live.add_result((dac.ch1, voltage), (dmm.v1, dmm.v1.get()))
```

The provider prepares a snapshot before registration and sets the run's
`source_package`. See `examples/qcodes_measurement.py` for a complete example.


`QuantifySnapshotProvider` derives the measurement ID and disk path from a
tuid. It retries when `dataset.hdf5` is not yet readable at the start of a run.
Install it with `pip install "qimchi-connect[quantify]"`.

```python
# Quantify
from qimchi_connect import QuantifySnapshotProvider, live_measurement

live = QuantifySnapshotProvider(tuid)
with live_measurement(live.measurement_id, live, disk_path=live.disk_path):
    MC.run(experiment_name)
```

`QCUtilsSnapshotProvider` reads the in-memory Zarr store used by a qcutils
sweep:

```python
# QCUtils
from qimchi_connect import QCUtilsSnapshotProvider, register_live_measurement

register_live_measurement(measurement_id, QCUtilsSnapshotProvider(memory_store))
```

See [Adding a framework](CONTRIBUTING.md#adding-a-framework) to support another
framework.

## Runnable examples

The examples do not require hardware. All three call
`qimchi_connect.live_measurement`; only the `snapshot` argument differs.

- [`examples/qcodes_measurement.py`](examples/qcodes_measurement.py) uses
  `QCoDeSSnapshotProvider` with a standard QCoDeS `Measurement` and `datasaver`.
  The provider manages the snapshot cache, locking, write flushing, and refresh
  rate. Install its dependency with `pip install qimchi-connect[examples]`.
- [`examples/quantify_measurement.py`](examples/quantify_measurement.py) uses
  `QuantifySnapshotProvider` with a two-dimensional Quantify
  `MeasurementControl` sweep. A discovery thread publishes the TUID that
  Quantify creates inside `run()`. Install its dependency with
  `pip install "qimchi-connect[quantify]"`.
- [`examples/generic_producer.py`](examples/generic_producer.py) uses an
  acquisition loop that updates a locked NumPy array and exposes it as an
  `xarray.Dataset`. Use this pattern for frameworks without a provider.

`tests/test_examples.py` runs all three scripts. When adding an example, add a
test for it in that file. `test_every_example_is_covered_by_a_test` checks this
requirement.

There is no separate qcutils script in this directory because qcutils already
includes higher-level examples based on `qcutils.measure.run` and `Sweep`.

## Repairing discovery records

Registration checks records marked as live. It marks a record as ended when
the endpoint remains unavailable after retries or no longer advertises the
recorded dataset. It then deletes ended records older than seven days.

Run the same maintenance manually with:

```console
qimchi-connect cleanup --retention-days 7
```

The discovery database is stored at `~/.qcutils/live_measurements.db` for
compatibility with existing qcutils and Qimchi installations.

## Runtime behavior

**One server per process.** The WebSocket server is process-wide. The first
`live_measurement` call starts it. Later calls use the same server. The `host`
and `port` arguments apply only to the call that starts the server. Each
registration advertises the active server.

**Snapshots are read-only.** Numeric and boolean variables are transferred as
raw bytes and received as read-only NumPy arrays. Call `.copy()` before
modifying a variable.

**The callback runs on the server thread.** It runs when a client requests data.
Use a lock and return a copy instead of a live buffer. See
`examples/generic_producer.py`. Use `QCoDeSSnapshotProvider` for QCoDeS.

## Development

```console
uv sync --extra dev --extra test --extra examples
uv run pre-commit install     # once per clone
uv run pytest
```

`ruff` is pinned to the same version in `pyproject.toml`,
`.pre-commit-config.yaml` and the CI `lint` job. Bump all three together.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch and release rules and for
instructions on adding framework support.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
