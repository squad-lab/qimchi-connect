# Contributing to Qimchi-Connect

## Getting set up

### Python support policy

We support the second-newest stable Python feature release. This balances timely
security and bug-fix support with enough time for our dependencies to stabilize
on each new Python version.

```console
uv sync --extra dev --extra test --extra examples
uv run pre-commit install     # once per clone
uv run pytest
```

The `dev` extra in `pyproject.toml`, `.pre-commit-config.yaml`, and the CI
`lint` job pin the same `ruff` version. Update all three together to keep local
and CI formatting consistent.

The Quantify tests use a fake `quantify_core` module, so the `quantify` extra is
not required to run them.

## Test strategy

The pytest suite covers the wire protocol, encoding, producer, server,
discovery registry, connection failures, concurrency, shutdown, framework
providers, and command-line tool. CI runs it on Python 3.13 and 3.14 on Linux
and Python 3.14 on Windows. All jobs must pass before merging, and every
pipeline reports coverage.

Use fake modules for frameworks unavailable in the test environment; see
`tests/test_quantify_provider.py`. Every script in `examples/` must have a test
that runs it.

Include a regression test with each bug fix and tests for each new feature.

## Branches and releases

- `main` is stable and accepts changes only through merges.
- Merge feature branches into `preview` first. All branches are linted and
  tested, and `preview` also builds the package. The `preview` branch is not
  tagged; only merges to `main` create releases.

On `main`, CI runs
[python-semantic-release](https://python-semantic-release.readthedocs.io). It
determines the next version from commits since the last tag, updates
`pyproject.toml` and `CHANGELOG.md`, creates a tag, publishes the package to
PyPI, and creates a GitLab Release with the CI job token.

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

Create a provider class when a callback is not sufficient. Common reasons are thread restrictions and framework-specific conventions.

**Thread-restricted handles.** The snapshot callback runs on the WebSocket
server thread. QCoDeS binds its SQLite connection to the measurement thread,
so reading it from the server thread raises `sqlite3.ProgrammingError`.
`QCoDeSSnapshotProvider` refreshes a locked cache on the measurement thread;
the callback reads that cache.

**Framework data conventions.** `QuantifySnapshotProvider` derives
`measurement_id`, `disk_path`, and `metadata` from a TUID.

A provider does not inherit from a base class. Implement the applicable
members:

| Member | Contract |
|---|---|
| `__call__(self) -> xr.Dataset` | Required. Return the run so far. Called from the server's thread, so it must be safe there. |
| `metadata` | Optional producer identification, used unless the caller passes its own. |
| `prepare(self)` | Optional. Called by `register_live_measurement` before publication to create the initial snapshot. |
| `measurement_id`, `disk_path` | Optional values exposed to the caller. They are not read automatically. |

Follow these rules:

1. **Import the framework lazily** inside `__init__`, and raise `ImportError`
   with the required extra. This keeps the framework optional.
2. **Do not raise from `__call__`.** Return the previous snapshot if a read
   fails while the framework is writing.
3. **Do not hold a handle open** across calls if the producer is still writing
   through it. Open, read, materialize, and close it in each call.

Then:

- Add an extra to `pyproject.toml` if the framework is not already a
  dependency. Document any dependency conflicts in `CONTRIBUTING.md`.
- Export the class from `qimchi_connect/__init__.py` and `__all__`.
- Add tests. If the framework cannot be installed in the test environment,
  inject a fake module as shown in `tests/test_quantify_provider.py`.
- Document it under [Supported frameworks](README.md#supported-frameworks).
- If you add a script to `examples/`, add a test that runs it.
  `test_every_example_is_covered_by_a_test` will fail otherwise.

## Style

- Use Google-style docstrings for public APIs and for internal code that needs
  explanation. Do not repeat the signature or an expressive test name.
- Comments should explain constraints or non-obvious decisions, not narrate
  the code or record its history.
- Full type hints.
- Run blocking work outside the event loop with `asyncio.to_thread`.
