import pytest

from qimchi_connect import server


@pytest.fixture(autouse=True)
def isolated_live_state():
    server.stop_live_server()
    server._PROVIDERS.clear()
    server._clear_snapshot_cache()
    yield
    server.stop_live_server()
    server._PROVIDERS.clear()
    server._clear_snapshot_cache()
