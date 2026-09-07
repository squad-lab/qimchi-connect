"""Configuration checks for the live WebSocket server."""

from __future__ import annotations

import logging

import pytest

from qimchi_connect import server


def test_snapshot_cache_ttl_uses_its_default_when_unset(monkeypatch):
    monkeypatch.delenv("QIMCHI_CONNECT_SNAPSHOT_TTL", raising=False)

    assert server._configured_snapshot_cache_ttl() == server.DEFAULT_SNAPSHOT_CACHE_TTL


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("0", 0.0), ("0.5", 0.5), (" 2 ", 2.0)],
)
def test_snapshot_cache_ttl_accepts_non_negative_seconds(
    monkeypatch, configured, expected
):
    monkeypatch.setenv("QIMCHI_CONNECT_SNAPSHOT_TTL", configured)

    assert server._configured_snapshot_cache_ttl() == expected


@pytest.mark.parametrize("configured", ["-1", "nan", "inf", "invalid"])
def test_invalid_snapshot_cache_ttl_warns_and_uses_the_default(
    monkeypatch, caplog, configured
):
    monkeypatch.setenv("QIMCHI_CONNECT_SNAPSHOT_TTL", configured)

    with caplog.at_level(logging.WARNING, logger="qimchi_connect.server"):
        result = server._configured_snapshot_cache_ttl()

    assert result == server.DEFAULT_SNAPSHOT_CACHE_TTL
    assert "Ignoring invalid QIMCHI_CONNECT_SNAPSHOT_TTL" in caplog.text
