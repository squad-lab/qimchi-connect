import xarray as xr

from qimchi_connect import producer, registry


def test_registration_runs_maintenance_and_closes_cleanly(monkeypatch):
    calls = []
    original_maintenance = registry.maintain_registry

    def maintain(*args, **kwargs):
        calls.append((args, kwargs))
        return original_maintenance(*args, **kwargs)

    monkeypatch.setattr(registry, "maintain_registry", maintain)
    dataset = xr.Dataset({"value": ("x", [1.0])}, coords={"x": [0]})

    registration = producer.register_live_measurement("generic-1", lambda: dataset)

    assert calls == [((), {"retention_days": 7})]
    assert registry.get_measurement("generic-1").live_status is True
    registration.close()
    assert registry.get_measurement("generic-1").live_status is False


class _RewritingProvider:
    append_only = False

    def __init__(self, dataset: xr.Dataset) -> None:
        self._dataset = dataset

    def __call__(self) -> xr.Dataset:
        return self._dataset


def _rows(measurement_id: str) -> dict:
    from qimchi_connect import client

    registration = producer.get_live_registration(measurement_id)
    dataset = client.open_live_measurement_sync(measurement_id, registration.ws_url)
    return dataset.encoding.get("qimchi_connect_rows", {})


def test_a_measurement_that_rewrites_rows_is_always_sent_whole():
    dataset = xr.Dataset({"value": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})

    with producer.live_measurement("rows", lambda: dataset):
        assert _rows("rows")["append_dim"] == "x"
    with producer.live_measurement("whole", lambda: dataset, append_only=False):
        assert _rows("whole") == {}


def test_a_provider_can_say_it_rewrites_rows():
    dataset = xr.Dataset({"value": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    provider = _RewritingProvider(dataset)

    with producer.live_measurement("provider", provider):
        assert _rows("provider") == {}
    with producer.live_measurement("override", provider, append_only=True):
        assert _rows("override")["append_dim"] == "x"
