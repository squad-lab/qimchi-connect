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
