"""
How a probe reports failure.

``reconcile_live_measurements`` is the only caller that has to act on a failed
probe, and all it gets is the exception -- so the exception has to say what
went wrong, and arrive within the budget the caller set.

"""

import socket
import time

import pytest

from qimchi_connect import client


def _closed_port() -> int:
    """
    Return a localhost port with nothing listening on it.

    Returns:
        int: Port number that was bound and released, so a connection to it is
            refused rather than accepted.

    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_failed_probe_says_why_it_failed():
    """
    ``reconcile_live_measurements`` logs the probe exception verbatim. An
    exception with no message leaves the log line reading
    "failed 2 probe(s): " with nothing after the colon.

    """
    url = f"ws://127.0.0.1:{_closed_port()}"

    with pytest.raises(Exception) as failure:
        client.list_live_measurements_sync(url, timeout=0.5, connect_timeout=0.5)

    assert str(failure.value), (
        f"{type(failure.value).__name__} carried no explanation of the failure"
    )


def test_a_probe_of_a_silent_endpoint_gives_up_within_its_budget():
    """
    A socket that accepts and then sends nothing matches a producer whose
    server thread is starved. The caller polls, so the probe has to return
    within its budget.

    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    url = f"ws://127.0.0.1:{listener.getsockname()[1]}"

    try:
        started = time.perf_counter()
        with pytest.raises(Exception):
            client.list_live_measurements_sync(url, timeout=0.5, connect_timeout=0.5)
        elapsed = time.perf_counter() - started
    finally:
        listener.close()

    assert elapsed < 2.0, f"probe overran its budget by {elapsed - 0.5:.2f}s"
