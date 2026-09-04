"""
Tests for the qimchi-connect command-line entry point.

The CLI is a shipped console script, so its argument wiring and its exit
status are part of the public surface even though the work itself lives in
qimchi_connect.registry.

"""

from __future__ import annotations

import pytest

from qimchi_connect import cli, registry


def test_cleanup_forwards_its_options_to_the_registry(monkeypatch, capsys):
    calls: list[dict] = []

    def fake_maintain(**kwargs):
        calls.append(kwargs)
        return registry.RegistryMaintenanceResult(("a", "b"), 3)

    monkeypatch.setattr(cli, "maintain_registry", fake_maintain)

    assert (
        cli.main(
            ["cleanup", "--retention-days", "2", "--timeout", "0.5", "--retries", "1"]
        )
        == 0
    )

    assert calls == [{"retention_days": 2, "timeout": 0.5, "retries": 1}]
    printed = capsys.readouterr().out
    assert "Marked stale: 2" in printed
    assert "Deleted expired: 3" in printed


def test_cleanup_defaults_match_the_registry_defaults(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        cli,
        "maintain_registry",
        lambda **kwargs: (
            calls.append(kwargs) or registry.RegistryMaintenanceResult((), 0)
        ),
    )

    cli.main(["cleanup"])

    assert calls == [{"retention_days": 7, "timeout": 1.0, "retries": 2}]


def test_cleanup_reports_the_registry_it_acted_on(monkeypatch, capsys, tmp_path):
    registry.configure_database(tmp_path / "named.db")
    monkeypatch.setattr(
        cli, "maintain_registry", lambda **_: registry.RegistryMaintenanceResult((), 0)
    )

    cli.main(["cleanup"])

    assert "named.db" in capsys.readouterr().out


def test_cleanup_runs_against_the_real_registry(capsys):
    """
    End to end, against the real registry rather than a stubbed maintainer.
    The CLI owes a successful run and a report; the counts themselves are
    reconciliation policy.

    """
    registry.register_measurement("gone", None, "ws://127.0.0.1:1", 1)

    assert cli.main(["cleanup", "--timeout", "0.05", "--retries", "1"]) == 0

    printed = capsys.readouterr().out
    assert "Marked stale:" in printed
    assert "Deleted expired: 0" in printed


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == 2


def test_version_is_reported_and_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])

    assert excinfo.value.code == 0
    assert "qimchi-connect" in capsys.readouterr().out


def test_the_version_falls_back_when_the_package_is_not_installed(monkeypatch):
    """A source checkout run without an install still has a working --version."""
    from importlib.metadata import PackageNotFoundError

    def missing(_name):
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(cli, "version", missing)

    assert cli._version() == "unknown"
