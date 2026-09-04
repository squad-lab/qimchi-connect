from pathlib import Path

import pytest

from qimchi_connect import registry, server


@pytest.fixture(autouse=True)
def isolated_live_state(tmp_path: Path):
    registry.configure_database(tmp_path / "live.db")
    server.stop_live_server()
    server._PROVIDERS.clear()
    server._clear_snapshot_cache()
    yield
    server.stop_live_server()
    server._PROVIDERS.clear()
    server._clear_snapshot_cache()
    registry.configure_database(None)
