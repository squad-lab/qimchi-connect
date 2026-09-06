import datetime
import sqlite3
import time

import pytest

from qimchi_connect import registry


def _register(measurement_id: str, endpoint: str = "ws://localhost:9000"):
    registry.register_measurement(measurement_id, None, endpoint, 9000)


def test_reconcile_ends_what_an_endpoint_disowns_but_not_what_it_cannot_reach():
    _register("present")
    _register("missing")
    _register("unreachable", "ws://localhost:9001")

    def probe(endpoint: str, timeout: float) -> list[str]:
        assert timeout == 0.25
        if endpoint.endswith("9001"):
            raise ConnectionRefusedError(endpoint)
        return ["present"]

    stale = registry.reconcile_live_measurements(
        timeout=0.25,
        retries=2,
        list_measurements=probe,
    )

    assert set(stale) == {"missing"}
    assert {record.measurement_id for record in registry.get_live_measurements()} == {
        "present",
        "unreachable",
    }
    assert registry.get_measurement("missing").ended_at is not None


def test_maintenance_keeps_reachable_old_live_rows_and_deletes_old_ended_rows():
    _register("long-running")
    _register("old-ended")
    old_timestamp = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
    ).isoformat()
    registry.end_measurement("old-ended", old_timestamp)

    result = registry.maintain_registry(
        retention_days=7,
        list_measurements=lambda _endpoint, _timeout: ["long-running"],
    )

    assert result.stale_measurement_ids == ()
    assert result.deleted_count == 1
    assert registry.get_measurement("long-running").live_status is True
    assert registry.get_measurement("old-ended") is None


def test_maintenance_marks_stale_before_retention_without_deleting_it():
    _register("crashed")

    result = registry.maintain_registry(
        retention_days=7,
        list_measurements=lambda _endpoint, _timeout: [],
    )

    assert result.stale_measurement_ids == ("crashed",)
    assert result.deleted_count == 0
    crashed = registry.get_measurement("crashed")
    assert crashed.live_status is False
    assert crashed.ended_at is not None


def test_the_probe_budget_covers_the_connection_as_well_as_the_response():
    """
    Registration reconciles by default, so an endpoint that is slow to accept
    a connection delays every measurement start until the budget is spent.

    """
    from qimchi_connect import client

    budgets: list[tuple[float, float]] = []
    original = client.send_request

    async def record(request, ws_url=client.DEFAULT_WS_URL, **kwargs):
        budgets.append((kwargs["timeout"], kwargs["connect_timeout"]))
        return await original(request, ws_url, **kwargs)

    client.send_request = record
    try:
        _register("unreachable", "ws://127.0.0.1:1")
        registry.reconcile_live_measurements(timeout=0.25, retries=1)
    finally:
        client.send_request = original

    assert budgets == [(0.25, 0.25)]


def test_reconcile_rejects_a_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        registry.reconcile_live_measurements(timeout=0.0)


def test_reconcile_rejects_fewer_than_one_retry():
    with pytest.raises(ValueError, match="retries"):
        registry.reconcile_live_measurements(retries=0)


def test_cleanup_rejects_a_negative_retention():
    with pytest.raises(ValueError, match="days"):
        registry.cleanup_old_measurements(days=-1)


def test_the_registry_round_trips_a_measurement_through_its_lifecycle():
    """Register, look up, move on disk, then end -- the qanary sequence."""
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    registry.register_measurement(
        "old", "/tmp/old.nc", "ws://localhost:9000", 9000, started
    )

    assert registry.get_measurement("old").fpath == "/tmp/old.nc"
    assert [r.measurement_id for r in registry.get_live_measurements()] == ["old"]

    registry.update_measurement_path("old", "/tmp/moved.nc")
    assert registry.get_measurement("old").fpath == "/tmp/moved.nc"

    registry.end_measurement("old", started)
    assert registry.get_measurement("old").live_status is False
    assert len(registry.get_all_measurements()) == 1
    assert registry.LiveMeasurement is registry.LiveMeasurement


def test_the_default_registry_lives_in_qimchis_home(monkeypatch, tmp_path):
    """
    Every producer publishes into the one file Qimchi reads, so the registry
    belongs in Qimchi's home rather than any single producer's.

    """
    monkeypatch.delenv("QIMCHI_HOME", raising=False)
    monkeypatch.setattr(registry.Path, "home", staticmethod(lambda: tmp_path))
    registry.configure_database(None)

    path = registry.get_database_path()

    assert path == tmp_path / ".qimchi" / "live_measurements.db"
    assert path.parent.is_dir()


def test_qimchi_home_moves_the_registry(monkeypatch, tmp_path):
    """
    ``QIMCHI_HOME`` is what Qimchi itself honours, so a moved home has to move
    the registry too -- otherwise the viewer and its producers disagree.

    """
    home = tmp_path / "elsewhere"
    monkeypatch.setenv("QIMCHI_HOME", str(home))
    registry.configure_database(None)

    path = registry.get_database_path()

    assert path == home / "live_measurements.db"
    assert path.parent.is_dir()


def test_qimchi_home_expands_the_user_directory(monkeypatch, tmp_path):
    """Producer and viewer must resolve a tilde-prefixed override equally."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("QIMCHI_HOME", "~/elsewhere")
    registry.configure_database(None)

    path = registry.get_database_path()

    assert path == tmp_path / "elsewhere" / "live_measurements.db"
    assert path.parent.is_dir()


def test_a_failed_statement_rolls_the_transaction_back():
    registry.init_database()

    with pytest.raises(sqlite3.IntegrityError):
        with registry._connection() as connection:
            connection.execute(
                """
                INSERT INTO live_measurements
                (measurement_id, fpath, ws_url, ws_port, live_status,
                 started_at, ended_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("kept", "", "ws://localhost:1", 1, True, "now", None),
            )
            connection.execute(
                "INSERT INTO live_measurements (measurement_id) VALUES (NULL)"
            )

    assert registry.get_measurement("kept") is None


def _unreachable(_endpoint: str, _timeout: float) -> list[str]:
    """
    Stand in for a producer that never replies.

    Args:
        _endpoint (str): Ignored endpoint URL.
        _timeout (float): Ignored probe budget.

    Raises:
        TimeoutError: Always.

    """
    raise TimeoutError


def test_an_unreachable_endpoint_survives_a_single_reconcile_round():
    """
    An unanswered probe says nothing about the producer: a measurement
    process starves its own server thread past any usable budget. Ending the
    record on that evidence unregisters a running measurement.

    """
    _register("running")

    stale = registry.reconcile_live_measurements(
        timeout=0.25,
        retries=2,
        list_measurements=_unreachable,
    )

    assert stale == ()
    assert registry.get_measurement("running").live_status is True


def test_an_endpoint_unreachable_past_the_grace_period_is_ended():
    """A producer that really is gone still has to leave the live list."""
    _register("crashed")

    assert (
        registry.reconcile_live_measurements(
            timeout=0.25,
            retries=1,
            unreachable_grace=0.3,
            list_measurements=_unreachable,
        )
        == ()
    )
    assert registry.get_measurement("crashed").live_status is True

    time.sleep(0.35)
    stale = registry.reconcile_live_measurements(
        timeout=0.25,
        retries=1,
        unreachable_grace=0.3,
        list_measurements=_unreachable,
    )

    assert stale == ("crashed",)
    crashed = registry.get_measurement("crashed")
    assert crashed.live_status is False
    assert crashed.ended_at is not None


def test_one_answer_restarts_the_grace_period():
    """
    Contention is bursty, so the window is measured from the last answer. A
    running total would accumulate unrelated misses and eventually end a live
    measurement.

    """
    _register("flaky")
    answers: list[list[str] | None] = [None, ["flaky"], None]

    def flaky(_endpoint: str, _timeout: float) -> list[str]:
        answer = answers.pop(0)
        if answer is None:
            raise TimeoutError
        return answer

    for _ in range(3):
        registry.reconcile_live_measurements(
            timeout=0.25,
            retries=1,
            unreachable_grace=0.3,
            list_measurements=flaky,
        )
        time.sleep(0.2)

    assert registry.get_measurement("flaky").live_status is True


def test_an_answering_endpoint_ends_a_measurement_it_does_not_list_at_once():
    """
    An endpoint that answers reports what it serves, so no grace period
    applies.

    """
    _register("finished")

    stale = registry.reconcile_live_measurements(
        timeout=0.25,
        retries=1,
        list_measurements=lambda _endpoint, _timeout: [],
    )

    assert stale == ("finished",)
    assert registry.get_measurement("finished").live_status is False


def test_registering_a_measurement_starts_it_with_a_clean_probe_record():
    """Ports get reused, so a fresh registration must not inherit misses."""
    _register("reused")
    registry.reconcile_live_measurements(
        timeout=0.25,
        retries=1,
        unreachable_grace=0.3,
        list_measurements=_unreachable,
    )
    time.sleep(0.35)

    _register("reused")

    assert (
        registry.reconcile_live_measurements(
            timeout=0.25,
            retries=1,
            unreachable_grace=0.3,
            list_measurements=_unreachable,
        )
        == ()
    )
    assert registry.get_measurement("reused").live_status is True


def test_reconcile_rejects_a_negative_grace_period():
    with pytest.raises(ValueError, match="unreachable_grace"):
        registry.reconcile_live_measurements(unreachable_grace=-1.0)
