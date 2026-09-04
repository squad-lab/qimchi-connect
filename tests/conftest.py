from pathlib import Path

import pytest

from qimchi_connect import producer, registry, server


@pytest.fixture(autouse=True)
def isolated_live_state(tmp_path: Path):
    registry.configure_database(tmp_path / "live.db")
    producer.stop_heartbeat()
    producer._REGISTRATIONS.clear()
    server.stop_live_server()
    server._PROVIDERS.clear()
    server._clear_snapshot_cache()
    yield
    producer.stop_heartbeat()
    producer._REGISTRATIONS.clear()
    server.stop_live_server()
    server._PROVIDERS.clear()
    server._clear_snapshot_cache()
    registry.configure_database(None)
