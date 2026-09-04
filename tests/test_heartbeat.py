"""
Producer liveness by heartbeat.

A consumer cannot tell a busy producer from a dead one by probing it: the probe
budget that would separate them is longer than the poll interval, and a
measurement process starves its own server thread for longer than either. So
the producer asserts that it is alive instead, and the registry query filters on
that assertion rather than on the result of a network round trip.

"""

from __future__ import annotations

import datetime
import time

import pytest
import xarray as xr

from qimchi_connect import producer, registry


def _register(measurement_id: str, endpoint: str = "ws://localhost:9000") -> None:
    """
    Add a live discovery row.

    Args:
        measurement_id (str): Identifier for the row.
        endpoint (str): Advertised WebSocket endpoint.

    """
    registry.register_measurement(measurement_id, None, endpoint, 9000)


def _explode(_endpoint: str, _timeout: float) -> list[str]:
    """
    Stand in for a probe that must never be reached.

    Args:
        _endpoint (str): Ignored endpoint URL.
        _timeout (float): Ignored probe budget.

    Raises:
        AssertionError: Always.

    """
    raise AssertionError("a beating producer must not be probed")


class TestTheRegistryQuery:
    def test_a_beating_producer_is_live_and_is_never_probed(self):
        """The heartbeat replaces the probe rather than adding to it."""
        _register("beating")
        registry.heartbeat("beating")

        live = registry.get_live_measurements()

        assert [record.measurement_id for record in live] == ["beating"]
        assert registry.reconcile_live_measurements(list_measurements=_explode) == ()

    def test_a_producer_that_stops_beating_leaves_the_live_list(self):
        _register("stopped")
        registry.heartbeat("stopped")
        time.sleep(0.1)

        assert registry.get_live_measurements(stale_after=0.05) == []

    def test_a_producer_that_stops_beating_is_ended_rather_than_left_behind(self):
        """
        Hiding the row from the query is not enough. It still reads
        live_status = 1, and cleanup only deletes ended rows, so it would
        never be removed.

        """
        _register("stopped")
        registry.heartbeat("stopped")
        time.sleep(0.1)

        stale = registry.reconcile_live_measurements(
            stale_after=0.05,
            list_measurements=_explode,
        )

        assert stale == ("stopped",)
        assert registry.get_measurement("stopped").live_status is False

    def test_a_producer_that_never_beats_is_still_probed(self):
        """
        A row can have no heartbeat: register_measurement writes none,
        publishing can opt out, and the heartbeat thread can die while the
        producer keeps serving. The endpoint decides in that case.

        """
        _register("legacy")
        probed: list[str] = []

        def probe(endpoint: str, _timeout: float) -> list[str]:
            probed.append(endpoint)
            return ["legacy"]

        assert registry.reconcile_live_measurements(list_measurements=probe) == ()
        assert probed == ["ws://localhost:9000"]
        assert [r.measurement_id for r in registry.get_live_measurements()] == [
            "legacy"
        ]

    def test_a_heartbeat_stamped_in_another_timezone_still_reads_as_fresh(self):
        """
        ISO-8601 sorts lexically only when every stamp carries the same
        offset. A fresh beat written west of UTC sorts below a UTC cutoff, so
        comparing the strings would read it as stale.

        """
        _register("west")
        west = datetime.timezone(datetime.timedelta(hours=-5))
        now = datetime.datetime.now(datetime.timezone.utc).astimezone(west)
        registry.heartbeat("west", when=now.isoformat())

        assert [r.measurement_id for r in registry.get_live_measurements()] == ["west"]


class TestTheProducerSide:
    def test_publishing_a_measurement_starts_its_heartbeat(self):
        dataset = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})

        registration = producer.register_live_measurement(
            "beating", lambda: dataset, heartbeat_interval=0.05
        )
        try:
            first = registry.get_measurement("beating").last_seen
            assert first is not None, "publishing did not stamp a first beat"

            time.sleep(0.2)
            assert registry.get_measurement("beating").last_seen != first
        finally:
            registration.close()

    def test_closing_one_of_several_registrations_keeps_the_rest_beating(self):
        """
        One thread beats for every open registration, so it stops only when
        the count reaches zero. Stopping on the first close would leave a
        running measurement unbeaten.

        """
        dataset = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})

        first = producer.register_live_measurement(
            "one", lambda: dataset, heartbeat_interval=0.05
        )
        second = producer.register_live_measurement(
            "two", lambda: dataset, heartbeat_interval=0.05
        )
        try:
            first.close()
            settled = registry.get_measurement("two").last_seen

            time.sleep(0.2)

            assert registry.get_measurement("two").last_seen != settled
        finally:
            second.close()

    def test_the_first_registration_sets_the_cadence_for_the_rest(self):
        """
        The thread is shared, so a later publish keeps the running interval
        rather than restarting on its own.

        """
        dataset = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})

        fast = producer.register_live_measurement(
            "fast", lambda: dataset, heartbeat_interval=0.05
        )
        slow = producer.register_live_measurement(
            "slow", lambda: dataset, heartbeat_interval=30.0
        )
        try:
            settled = registry.get_measurement("slow").last_seen

            time.sleep(0.2)

            assert registry.get_measurement("slow").last_seen != settled
        finally:
            slow.close()
            fast.close()

    def test_a_heartbeat_slower_than_the_staleness_window_is_warned_about(self, caplog):
        """
        Beats further apart than the staleness window leave the measurement
        absent from the live list between them.

        """
        dataset = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})

        with caplog.at_level("WARNING", logger="qimchi_connect.producer"):
            registration = producer.register_live_measurement(
                "sluggish",
                lambda: dataset,
                heartbeat_interval=registry.DEFAULT_STALE_AFTER + 1,
            )
        registration.close()

        assert "staleness window" in caplog.text

    def test_a_default_heartbeat_is_not_warned_about(self, caplog):
        dataset = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})

        with caplog.at_level("WARNING", logger="qimchi_connect.producer"):
            registration = producer.register_live_measurement("steady", lambda: dataset)
        registration.close()

        assert "staleness window" not in caplog.text

    def test_closing_the_last_registration_stops_the_heartbeat(self):
        """A closed measurement must not keep asserting that it is running."""
        dataset = xr.Dataset({"v": ("x", [1.0])}, coords={"x": [0]})

        registration = producer.register_live_measurement(
            "ending", lambda: dataset, heartbeat_interval=0.05
        )
        registration.close()
        registry.register_measurement("ending", None, "ws://localhost:9000", 9000)
        registry.heartbeat("ending")
        settled = registry.get_measurement("ending").last_seen

        time.sleep(0.2)

        assert registry.get_measurement("ending").last_seen == settled


def test_reconcile_rejects_a_negative_staleness_window():
    with pytest.raises(ValueError, match="stale_after"):
        registry.reconcile_live_measurements(stale_after=-1.0)


def test_a_heartbeat_for_an_unknown_measurement_is_a_no_op():
    """
    The heartbeat thread can beat for a row cleanup already deleted, so this
    is a no-op rather than an error it has to catch.

    """
    registry.heartbeat("never-registered")

    assert registry.get_measurement("never-registered") is None
