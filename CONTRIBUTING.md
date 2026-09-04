# Contributing to qimchi-connect

## Getting set up

```console
uv sync --extra dev --extra test --extra examples
uv run pre-commit install     # once per clone
uv run pytest
```

`ruff` is pinned in the `dev` extra in `pyproject.toml`,
`.pre-commit-config.yaml`, and the CI `lint` job. Use the same version in all
three places. Different versions can produce different formatting results.

The Quantify tests use a fake `quantify_core` module, so the `quantify` extra is
not required to run them.

## Branches and releases

- `main` is stable and accepts changes only through merges.
- Merge feature branches into `preview` first. All branches are linted and
  tested. The `preview` branch also builds the package. It is not tagged. Only
  merges to `main` create releases.

CI runs [python-semantic-release](https://python-semantic-release.readthedocs.io)
on `main`. It reads commits since the last tag, updates the version in
`pyproject.toml`, updates `CHANGELOG.md`, creates a tag, and publishes the
package to PyPI. The same pipeline creates a GitLab Release for that tag using
the CI job token.

| Prefix | Release |
|---|---|
| `fix:` | patch |
| `feat:` | minor |
| `BREAKING CHANGE:` in the body | major |
| anything else (`chore:`, `docs:`, `test:`, `refactor:`) | none |

Use `fix:`, `feat:`, or `BREAKING CHANGE:` for changes that require a release.
Do not edit the `version` field. semantic-release manages it.

## Adding a framework

A framework can use a callback that returns an `xarray.Dataset`:

```python
with live_measurement("run-42", lambda: build_current_dataset()):
    acquire()
```

Create a provider class when a callback is not sufficient. Common reasons are
thread restrictions and framework-specific conventions.

**The framework's handle cannot be read from the server's thread.** The
snapshot callback runs on the WebSocket server's thread whenever a client asks
for data. QCoDeS holds a thread-affine SQLite connection, so reading it there
raises `sqlite3.ProgrammingError`. `QCoDeSSnapshotProvider` refreshes a cached
snapshot on the measurement thread. The callback reads that cache under a
lock.

**The framework has data conventions.** `QuantifySnapshotProvider` derives
`measurement_id`, `disk_path`, and `metadata` from a tuid.

A provider does not inherit from a base class. Implement the applicable
members:

| Member | Contract |
|---|---|
| `__call__(self) -> xr.Dataset` | Required. Return the run so far. Called from the server's thread, so it must be safe there. |
| `metadata` | Optional. Producer identification, used unless the caller passes its own. |
| `prepare(self)` | Optional. Called by `register_live_measurement` before publishing, so a client that connects immediately sees the run so far. |
| `measurement_id`, `disk_path` | Optional values for the caller. They are not read automatically. |

Follow these rules:

1. **Import the framework lazily**, inside `__init__`, and raise `ImportError`
   with the name of the required extra. This keeps the rest of the package
   usable without the framework.
2. **Do not raise from `__call__`.** If a read fails while the framework is
   writing, return the previous snapshot.
3. **Do not hold a handle open** across calls if the producer is still writing
   through it. Open, read, materialize, and close it in each call.

Then:

- Add the extra to `pyproject.toml` if the framework is not already a
  dependency, and say in `CONTRIBUTING.md` if it conflicts with another.
- Export the class from `qimchi_connect/__init__.py` and `__all__`.
- Add tests. If the framework cannot be installed in the test environment,
  inject a fake module as shown in `tests/test_quantify_provider.py`.
- Document it under [Supported frameworks](README.md#supported-frameworks).
- If you add a script to `examples/`, add a test that runs it.
  `test_every_example_is_covered_by_a_test` will fail otherwise.

## Style

- Google-style docstrings with `Args:`, `Returns:` and `Raises:`, on every
  function and class.
- Docstrings and comments describe what the code does and the constraints it
  works under. Do not include change history.
- Full type hints.
- Run blocking work outside the event loop with `asyncio.to_thread`.
