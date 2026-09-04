"""
Tests for publication lifecycle and client error surfacing.

register_live_measurement ties together three pieces of state -- the snapshot
provider, the WebSocket server and the discovery row -- and any of them can
fail, or a handle can be used after it is closed. On the client side, a
server-reported failure has to become an exception rather than an empty result
the caller mistakes for real data.

"""

from __future__ import annotations

import asyncio

import pytest
import xarray as xr

from qimchi_connect import client, producer, registry, server


@pytest.fixture
def dataset() -> xr.Dataset:
    return xr.Dataset({"signal": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})


class TestRegistrationHandle:
    def test_the_advertised_disk_path_can_be_updated(self, dataset, tmp_path):
        final = tmp_path / "run.nc"

        with producer.live_measurement("run", lambda: dataset) as registration:
            registration.update_disk_path(final)

            assert registration.disk_path == str(final)
            assert registry.get_measurement("run").fpath == str(final)

    def test_updating_a_closed_registration_is_an_error(self, dataset, tmp_path):
        registration = producer.register_live_measurement("run", lambda: dataset)
        registration.close()

        with pytest.raises(RuntimeError, match="closed"):
            registration.update_disk_path(tmp_path / "run.nc")

    def test_closing_twice_is_harmless(self, dataset):
        registration = producer.register_live_measurement("run", lambda: dataset)

        registration.close()
        registration.close()

        assert registry.get_measurement("run").live_status is False

    def test_the_context_manager_closes_on_an_exception(self, dataset):
        with pytest.raises(ValueError):
            with producer.live_measurement("run", lambda: dataset):
                raise ValueError("measurement failed")

        assert registry.get_measurement("run").live_status is False
        assert server._provider_entry("run") is None

    def test_republishing_an_id_supersedes_the_previous_handle(self, dataset):
        first = producer.register_live_measurement("run", lambda: dataset)
        second = producer.register_live_measurement("run", lambda: dataset)

        first.close()

        assert server._provider_entry("run") is not None
        assert registry.get_measurement("run").live_status is True
        second.close()
        assert registry.get_measurement("run").live_status is False

    def test_a_registration_advertises_the_running_server(self, dataset):
        with producer.live_measurement("run", lambda: dataset) as registration:
            assert registration.ws_port == server.get_server_port()
            assert registration.ws_url.endswith(str(registration.ws_port))


class TestRegistrationFailure:
    def test_a_failed_discovery_write_unregisters_the_provider(
        self, dataset, monkeypatch
    ):
        """Otherwise the server mis-advertises a measurement."""

        def refuse(*args, **kwargs):
            raise sqlite_error()

        def sqlite_error():
            return RuntimeError("registry is read-only")

        monkeypatch.setattr(registry, "register_measurement", refuse)

        with pytest.raises(RuntimeError, match="read-only"):
            producer.register_live_measurement("run", lambda: dataset)

        assert server._provider_entry("run") is None

    def test_a_server_that_will_not_start_unregisters_the_provider(
        self, dataset, monkeypatch
    ):
        monkeypatch.setattr(server, "is_server_running", lambda: False)
        monkeypatch.setattr(server, "start_live_server", lambda *a, **k: False)

        with pytest.raises(RuntimeError, match="failed to start"):
            producer.register_live_measurement("run", lambda: dataset)

        assert server._provider_entry("run") is None

    def test_maintenance_can_be_skipped(self, dataset, monkeypatch):
        called: list[int] = []
        monkeypatch.setattr(registry, "maintain_registry", lambda **_: called.append(1))

        producer.register_live_measurement("run", lambda: dataset, maintain=False)

        assert called == []


class TestClientErrorSurfacing:
    @pytest.fixture
    def endpoint(self, dataset) -> str:
        server.register_snapshot_provider("run", lambda: dataset)
        assert server.start_live_server(port=0)
        return f"ws://localhost:{server.get_server_port()}"

    def test_an_unknown_measurement_raises_rather_than_returning_nothing(
        self, endpoint
    ):
        with pytest.raises(RuntimeError, match="absent"):
            asyncio.run(client.open_live_measurement("absent", endpoint))

    def test_a_failed_info_request_raises(self, endpoint):
        with pytest.raises(RuntimeError, match="measurement info"):
            asyncio.run(client.get_measurement_info("absent", endpoint))

    def test_a_failed_data_request_raises(self, endpoint):
        with pytest.raises(RuntimeError, match="measurement data"):
            asyncio.run(client.get_measurement_data("absent", ["signal"], endpoint))

    def test_listing_an_empty_server_returns_nothing(self):
        """An idle server has no measurements; that is not an error."""
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        assert client.list_live_measurements_sync(endpoint) == []

    def test_the_source_metadata_is_attached_to_the_loaded_dataset(self, dataset):
        server.register_snapshot_provider(
            "described", lambda: dataset, {"source_package": "test"}
        )
        assert server.start_live_server(port=0)
        endpoint = f"ws://localhost:{server.get_server_port()}"

        loaded = client.open_live_measurement_sync("described", endpoint)

        assert loaded.encoding["qimchi_connect_source"] == {"source_package": "test"}


class TestServerLifecycle:
    def test_starting_an_already_running_server_is_a_no_op(self):
        assert server.start_live_server(port=0)
        port = server.get_server_port()

        assert server.start_live_server(port=0)

        assert server.get_server_port() == port

    def test_a_stopped_server_reports_no_host_or_port(self):
        assert server.start_live_server(port=0)

        server.stop_live_server()

        assert server.is_server_running() is False
        assert server.get_server_port() == 0
        assert server.get_server_host() == ""

    def test_stopping_a_server_that_never_started_is_harmless(self):
        server.stop_live_server()

        assert server.is_server_running() is False
