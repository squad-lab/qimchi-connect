# Qimchi-Connect

[![pipeline](https://gitlab.com/squad-lab/qimchi-connect/badges/main/pipeline.svg?ignore_skipped=true&key_text=pipeline&key_width=60)](https://gitlab.com/squad-lab/qimchi-connect/-/pipelines?ref=main)
[![tests](https://gitlab.com/squad-lab/qimchi-connect/badges/main/pipeline.svg?job=pytest%3A%20%5B3.13%5D&ignore_skipped=true&key_text=tests&key_width=40)](https://gitlab.com/squad-lab/qimchi-connect/-/pipelines?ref=main)
[![coverage](https://gitlab.com/squad-lab/qimchi-connect/badges/main/coverage.svg?key_text=coverage&key_width=64)](https://gitlab.com/squad-lab/qimchi-connect/-/jobs)
[![latest release](https://gitlab.com/squad-lab/qimchi-connect/-/badges/release.svg?key_text=release&key_width=54)](https://gitlab.com/squad-lab/qimchi-connect/-/releases)
[![PyPI version](https://img.shields.io/pypi/v/qimchi-connect)](https://pypi.org/project/qimchi-connect/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23536727.svg)](https://doi.org/10.5281/zenodo.23536727)

Qimchi-Connect lets Qimchi display an `xarray.Dataset` while a measurement is
still running. It works with any acquisition code that can return a consistent
snapshot, and includes providers for QCoDeS, Qanary, and Quantify.

## Installation

Install the latest release from PyPI:

```console
pip install qimchi-connect
```

With `uv`:

```console
uv add qimchi-connect
```

Install the current `preview` branch with:

```console
pip install "qimchi-connect @ git+https://gitlab.com/squad-lab/qimchi-connect.git@preview"
```

Or, with `uv`:

```console
uv add "qimchi-connect @ git+https://gitlab.com/squad-lab/qimchi-connect.git@preview"
```

Qimchi-Connect requires Python 3.13 or later. CI tests Python 3.13 and 3.14.

Optional dependencies are grouped into extras:

| Extra | Adds | For |
|---|---|---|
| `examples` | `qcodes` | `QCoDeSSnapshotProvider` and `examples/qcodes_measurement.py` |
| `quantify` | `quantify-core` | `QuantifySnapshotProvider` and `examples/quantify_measurement.py` |
| `dev` | pre-commit, ruff, pytest | Development |
| `test` | pytest, pytest-cov | Tests |

## Publishing a measurement

Pass `live_measurement` a callback that returns the latest dataset. The callback
runs when a client asks for data and may run on another thread, so protect
shared state with the same lock used by the acquisition code. Return a new
snapshot: the server loads it into memory and closes it after reading.

```python
from qimchi_connect import live_measurement

with live_measurement(
    measurement_id="experiment-42",
    snapshot=lambda: current_dataset.copy(deep=True),
    disk_path="data/experiment-42.nc",
):
    perform_measurement()
```

## Following a measurement incrementally

If every data variable grows along the same leading dimension, a client can
fetch only the rows added since its previous request. Use the row count from
the first response as the next offset:

```python
from qimchi_connect import open_live_measurement_sync

snapshot = open_live_measurement_sync("experiment-42", ws_url)
progress = snapshot.encoding.get("qimchi_connect_rows")

if progress is not None:
    delta = open_live_measurement_sync(
        "experiment-42",
        ws_url,
        since_rows=progress["rows_written"],
    )
```

`delta` contains the new rows but is not merged into the earlier dataset. Its
`encoding["qimchi_connect_rows"]` contains the append dimension, returned
offset, written row count, and allocated size. Older servers ignore
`since_rows`, and datasets without a shared leading dimension are always sent
in full.

## Supported frameworks

Framework providers handle details that do not fit a plain callback, such as
thread-bound database handles or framework-specific paths and metadata.

The WebSocket server runs on a background thread. QCoDeS, however, binds its
SQLite connection to the measurement thread and buffers writes. Reading a
`datasaver` directly from the server can therefore raise
`sqlite3.ProgrammingError` or return stale data.

`QCoDeSSnapshotProvider` handles locking, buffered writes, snapshot caching,
and refresh rate limiting:

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

The provider creates an initial snapshot before registration and sets the
run's `source_package`. See `examples/qcodes_measurement.py` for a complete
example.

`QuantifySnapshotProvider` derives the measurement ID and disk path from a
TUID. It retains the previous snapshot when `dataset.hdf5` is temporarily
unreadable.
Install it with `pip install "qimchi-connect[quantify]"`.

```python
# Quantify
from qimchi_connect import QuantifySnapshotProvider, live_measurement

live = QuantifySnapshotProvider(tuid)
with live_measurement(live.measurement_id, live, disk_path=live.disk_path):
    MC.run(experiment_name)
```

`QanarySnapshotProvider` reads the in-memory Zarr store used by a Qanary
sweep:

```python
# Qanary
from qimchi_connect import QanarySnapshotProvider, register_live_measurement

register_live_measurement(measurement_id, QanarySnapshotProvider(memory_store))
```

In Qimchi-Connect 0.1.0 and earlier, Qanary was called qcutils and the provider
was named `QCUtilsSnapshotProvider`. Those names are no longer supported.

See [Adding a framework](CONTRIBUTING.md#adding-a-framework) to support another
framework.

## Runnable examples

All examples run without hardware:

- [`examples/qcodes_measurement.py`](examples/qcodes_measurement.py) uses
  `QCoDeSSnapshotProvider` with a standard QCoDeS `Measurement` and `datasaver`.
  The provider handles locking, buffered writes, snapshot caching, and refresh
  rate limiting. Install its dependency with
  `pip install qimchi-connect[examples]`.
- [`examples/quantify_measurement.py`](examples/quantify_measurement.py) uses
  `QuantifySnapshotProvider` with a two-dimensional Quantify
  `MeasurementControl` sweep. A discovery thread publishes the TUID that
  Quantify creates inside `run()`. Install its dependency with
  `pip install "qimchi-connect[quantify]"`.
- [`examples/generic_producer.py`](examples/generic_producer.py) uses an
  acquisition loop that updates a locked NumPy array and exposes it as an
  `xarray.Dataset`. Use this pattern for frameworks without a provider.

`tests/test_examples.py` runs each script. Add a matching test when adding an
example.

Qanary already provides higher-level examples based on `qanary.measure.run`
and `Sweep`, so this repository does not include a separate Qanary script.

## Repairing discovery records

Before registering a measurement, Qimchi-Connect checks existing live records.
Records are ended when their heartbeat expires, their endpoint no longer lists
the measurement, or the endpoint remains unreachable past the grace period.
Ended records are deleted after seven days.

Run the same maintenance manually with:

```console
qimchi-connect cleanup --retention-days 7
```

The discovery database is `$QIMCHI_HOME/live_measurements.db`, which defaults
to `~/.qimchi/live_measurements.db`. All producers write to the same database
that Qimchi reads.

Version 0.1.0 and earlier used `~/.qcutils/live_measurements.db`. That database
is not migrated because it contains only live and recently ended runs. Upgrade
Qimchi and its producers together, then remove the old `~/.qcutils` directory.

## Runtime behavior

**One server per process.** The first `live_measurement` call starts the
WebSocket server; later calls reuse it. Only the first call's `host` and `port`
take effect.

**Snapshots are read-only.** Numeric and boolean variables are transferred as
raw bytes and received as read-only NumPy arrays. Call `.copy()` before
modifying a variable.

**Callbacks run on the server thread.** Protect shared data with a lock and
return a copy, not a live buffer. See `examples/generic_producer.py`. QCoDeS
code should use `QCoDeSSnapshotProvider`.

**Snapshot caching is configurable.** By default, requests for the same
measurement reuse a serialized snapshot for 0.25 seconds. Set
`QIMCHI_CONNECT_SNAPSHOT_TTL` to a non-negative number of seconds; zero disables
cache hits. Invalid values emit a warning and use the default.

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

## Citing Qimchi-Connect

Please cite Qimchi-Connect when using it in your work:

```bibtex
@misc{qimchi-connect,
  author       = {Shivottam, Jyotirmaya and Anupam, Spandan and Mourik, Vincent},
  title        = {Qimchi-Connect: Publish and discover live xarray datasets in Qimchi},
  year         = {2026},
  note         = {Version 0.4.0, software},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23536727},
  url          = {https://doi.org/10.5281/zenodo.23536727}
}
```

This DOI covers all archived releases. To cite a specific version, use its DOI
from the [Zenodo record](https://doi.org/10.5281/zenodo.23536727).
[CITATION.cff](CITATION.cff) has the same information in machine-readable form.

## License

Qimchi-Connect is free software under the [AGPL-3.0-or-later](LICENSE).
Dependencies are installed separately under their own licenses.

Copyright © 2026 Forschungszentrum Jülich GmbH, PGI-11.
