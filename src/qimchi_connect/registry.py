"""
SQLite discovery registry for live measurement producers and consumers.

"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

_DATABASE_PATH: Path | None = None
_DATABASE_LOCK = threading.RLock()

# How long an endpoint has to stay unanswerable before its measurements are
# treated as gone.
DEFAULT_UNREACHABLE_GRACE = 60.0

# How often a producer stamps the registry to say it is still running, and how
# long a consumer waits before treating it as stopped. The window spans three
# missed beats, so a producer briefly too busy to answer a probe is not
# mistaken for one that has exited.
DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_STALE_AFTER = 3 * DEFAULT_HEARTBEAT_INTERVAL


@dataclass(slots=True)
class LiveMeasurement:
    """
    One row of the live discovery registry.

    Field and column names match the qcutils schema the table was first
    written with, so an existing ``live_measurements.db`` keeps working.

    """

    measurement_id: str
    fpath: str
    ws_url: str
    ws_port: int
    live_status: bool
    started_at: str
    ended_at: str | None = None
    last_seen: str | None = None


@dataclass(frozen=True, slots=True)
class RegistryMaintenanceResult:
    """
    Outcome of one :func:`maintain_registry` pass.

    """

    stale_measurement_ids: tuple[str, ...]
    deleted_count: int


def get_database_path() -> Path:
    """
    Return the live registry path, creating its parent directory if needed.

    Returns:
        Path: SQLite database path.

    """
    global _DATABASE_PATH
    if _DATABASE_PATH is None:
        directory = Path.home() / ".qcutils"
        directory.mkdir(parents=True, exist_ok=True)
        _DATABASE_PATH = directory / "live_measurements.db"
    return _DATABASE_PATH


def configure_database(path: str | Path | None) -> None:
    """
    Override the live registry path for the current process.

    Args:
        path (str | Path | None): New path, or ``None`` to restore the default.

    """
    global _DATABASE_PATH
    _DATABASE_PATH = Path(path) if path is not None else None


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """
    Open a serialized transactional database connection.

    Yields:
        sqlite3.Connection: Connection configured with named rows.

    """
    with _DATABASE_LOCK:
        database_path = get_database_path()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(database_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def init_database() -> None:
    """
    Create the live registry table and index if they do not exist.

    """
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_measurements (
                measurement_id TEXT PRIMARY KEY,
                fpath TEXT NOT NULL,
                ws_url TEXT NOT NULL,
                ws_port INTEGER NOT NULL,
                live_status BOOLEAN NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_status
            ON live_measurements(live_status)
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(live_measurements)")
        }
        if "missed_since" not in columns:
            connection.execute(
                "ALTER TABLE live_measurements ADD COLUMN missed_since TEXT"
            )
        if "last_seen" not in columns:
            connection.execute(
                "ALTER TABLE live_measurements ADD COLUMN last_seen TEXT"
            )


def register_measurement(
    measurement_id: str,
    disk_path: str | Path | None,
    ws_url: str,
    ws_port: int,
    started_at: str | None = None,
) -> None:
    """
    Register or replace a live measurement discovery record.

    Args:
        measurement_id (str): Stable measurement identifier.
        disk_path (str | Path | None): Optional persisted measurement location.
        ws_url (str): WebSocket endpoint advertising the measurement.
        ws_port (int): WebSocket TCP port.
        started_at (str | None): ISO timestamp, defaulting to current UTC time.

    """
    init_database()
    timestamp = started_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO live_measurements
            (measurement_id, fpath, ws_url, ws_port, live_status, started_at, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                measurement_id,
                str(disk_path or ""),
                ws_url,
                ws_port,
                True,
                timestamp,
                None,
            ),
        )


def end_measurement(measurement_id: str, ended_at: str | None = None) -> None:
    """
    Mark a discovery record as no longer live.

    Args:
        measurement_id (str): Dataset identifier to update.
        ended_at (str | None): ISO timestamp, defaulting to current UTC time.

    """
    init_database()
    timestamp = ended_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            """
            UPDATE live_measurements
            SET live_status = ?, ended_at = ?
            WHERE measurement_id = ?
            """,
            (False, timestamp, measurement_id),
        )


def heartbeat(measurement_id: str, when: str | None = None) -> None:
    """
    Record that a producer is still running.

    Called on a timer by the process that owns the measurement. A row whose
    stamp stops advancing is treated as gone.

    Args:
        measurement_id (str): Identifier to stamp.
        when (str | None): ISO timestamp, defaulting to current UTC time.

    """
    init_database()
    timestamp = when or datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            "UPDATE live_measurements SET last_seen = ? WHERE measurement_id = ?",
            (timestamp, measurement_id),
        )


def update_measurement_path(measurement_id: str, disk_path: str | Path) -> None:
    """
    Update the persisted path for a discovery record.

    Args:
        measurement_id (str): Dataset identifier to update.
        disk_path (str | Path): New persisted measurement location.

    """
    init_database()
    with _connection() as connection:
        connection.execute(
            """
            UPDATE live_measurements
            SET fpath = ?
            WHERE measurement_id = ?
            """,
            (str(disk_path), measurement_id),
        )


def _row_to_record(row: sqlite3.Row) -> LiveMeasurement:
    """
    Convert a database row into a discovery record.

    Args:
        row (sqlite3.Row): Named SQLite row.

    Returns:
        LiveMeasurement: Converted registry record.

    """
    return LiveMeasurement(
        measurement_id=row["measurement_id"],
        fpath=row["fpath"],
        ws_url=row["ws_url"],
        ws_port=row["ws_port"],
        live_status=bool(row["live_status"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        last_seen=row["last_seen"],
    )


# Freshness is defined here and read by every caller through _live_records.
# julianday() is used rather than a string comparison or datetime(): ISO-8601
# sorts lexically only when every stamp carries the same offset, so a fresh
# beat written west of UTC would sort below a UTC cutoff, and datetime()
# truncates to whole seconds, which widens any window shorter than that.
# julianday() normalises the offset and keeps sub-second resolution.
_LIVENESS_SQL = """
    SELECT measurement_id, fpath, ws_url, ws_port, live_status,
           started_at, ended_at, last_seen,
           CASE
               WHEN last_seen IS NULL THEN 'unknown'
               WHEN julianday(last_seen) >= julianday(?) THEN 'beating'
               ELSE 'stopped'
           END AS liveness
    FROM live_measurements
    WHERE live_status = ?
    ORDER BY started_at DESC
"""


def _live_records(stale_after: float) -> list[tuple[LiveMeasurement, str]]:
    """
    Return every record marked live, each tagged with how it reports liveness.

    Args:
        stale_after (float): Seconds a heartbeat may go unrefreshed before the
            producer is treated as gone.

    Returns:
        list[tuple[LiveMeasurement, str]]: Records paired with ``"beating"``,
            ``"stopped"``, or ``"unknown"`` for a producer that never stamped
            one.

    Raises:
        ValueError: If ``stale_after`` is negative.

    """
    if stale_after < 0:
        raise ValueError("stale_after must not be negative")

    init_database()
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=stale_after
    )
    with _connection() as connection:
        rows = connection.execute(_LIVENESS_SQL, (cutoff.isoformat(), True)).fetchall()
    return [(_row_to_record(row), row["liveness"]) for row in rows]


def get_live_measurements(
    *, stale_after: float = DEFAULT_STALE_AFTER
) -> list[LiveMeasurement]:
    """
    Return records currently marked live whose producer still looks alive.

    Args:
        stale_after (float): Seconds a heartbeat may go unrefreshed before the
            record is left out. Records from a producer that never stamps one
            are always included; reconciliation probes those instead.

    Returns:
        list[LiveMeasurement]: Live records ordered newest first.

    Raises:
        ValueError: If ``stale_after`` is negative.

    """
    return [
        record
        for record, liveness in _live_records(stale_after)
        if liveness != "stopped"
    ]


def get_all_measurements() -> list[LiveMeasurement]:
    """
    Return all discovery records.

    Returns:
        list[LiveMeasurement]: All records ordered newest first.

    """
    init_database()
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT measurement_id, fpath, ws_url, ws_port, live_status,
                   started_at, ended_at, last_seen
            FROM live_measurements
            ORDER BY started_at DESC
            """
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def get_measurement(measurement_id: str) -> LiveMeasurement | None:
    """
    Return one discovery record by identifier.

    Args:
        measurement_id (str): Dataset identifier to resolve.

    Returns:
        LiveMeasurement | None: Matching record, or ``None`` if absent.

    """
    init_database()
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT measurement_id, fpath, ws_url, ws_port, live_status,
                   started_at, ended_at, last_seen
            FROM live_measurements
            WHERE measurement_id = ?
            """,
            (measurement_id,),
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def cleanup_old_measurements(days: int = 7) -> int:
    """
    Delete ended records older than the retention period.

    Args:
        days (int): Minimum age in whole days for deletion.

    Returns:
        int: Number of deleted records.

    Raises:
        ValueError: If ``days`` is negative.

    """
    if days < 0:
        raise ValueError("days must not be negative")

    init_database()
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=days
    )
    with _connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM live_measurements
            WHERE live_status = ?
              AND ended_at IS NOT NULL
              AND datetime(ended_at) < datetime(?)
            """,
            (False, cutoff.isoformat()),
        )
        return cursor.rowcount


def _note_unreachable(measurement_id: str, now: datetime.datetime) -> float:
    """
    Record that an endpoint went unanswered and report how long it has been.

    The first miss stamps the row and later ones leave it, so the elapsed time
    is measured from the last answer.

    Args:
        measurement_id (str): Identifier whose endpoint failed to answer.
        now (datetime.datetime): Timestamp of the failed round.

    Returns:
        float: Seconds the endpoint has been unanswerable, zero on the first
            miss or when the stored stamp cannot be read.

    """
    with _connection() as connection:
        row = connection.execute(
            "SELECT missed_since FROM live_measurements WHERE measurement_id = ?",
            (measurement_id,),
        ).fetchone()
        stamp = row["missed_since"] if row is not None else None
        if stamp is None:
            connection.execute(
                "UPDATE live_measurements SET missed_since = ? WHERE measurement_id = ?",
                (now.isoformat(), measurement_id),
            )
            return 0.0

    try:
        since = datetime.datetime.fromisoformat(stamp)
    except ValueError:
        return 0.0
    return (now - since).total_seconds()


def _note_reachable(measurement_id: str) -> None:
    """
    Clear an endpoint's miss stamp after it answers.

    Args:
        measurement_id (str): Identifier whose endpoint answered.

    """
    with _connection() as connection:
        connection.execute(
            "UPDATE live_measurements SET missed_since = NULL WHERE measurement_id = ?",
            (measurement_id,),
        )


def reconcile_live_measurements(
    *,
    timeout: float = 1.0,
    retries: int = 2,
    stale_after: float = DEFAULT_STALE_AFTER,
    unreachable_grace: float = DEFAULT_UNREACHABLE_GRACE,
    list_measurements: Callable[[str, float], list[str]] | None = None,
) -> tuple[str, ...]:
    """
    End live records whose producer has gone.

    A producer is gone if its heartbeat stopped, or -- for a producer that
    writes none -- if its endpoint stops listing the measurement, or stays
    unanswerable past the grace period.

    Args:
        timeout (float): Budget in seconds for each endpoint attempt, applied
            to the connection and to the response separately.
        retries (int): Number of endpoint attempts before declaring failure.
        stale_after (float): Seconds a producer's heartbeat may go unrefreshed
            before its measurements are ended. Producers that stamp one are
            never probed; the rest fall through to the probe below.
        unreachable_grace (float): Seconds an endpoint may stay unanswerable
            before the measurements it carried are ended. Applies only to
            producers with no heartbeat, and only while the endpoint does not
            answer; an answer settles the question immediately.
        list_measurements (Callable[[str, float], list[str]] | None): Optional probe
            used for testing or custom transports.

    Returns:
        tuple[str, ...]: Identifiers newly marked as ended.

    Raises:
        ValueError: If ``timeout`` is not positive, ``retries`` is less than
            one, or ``stale_after`` or ``unreachable_grace`` is negative.

    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if retries < 1:
        raise ValueError("retries must be at least one")
    if unreachable_grace < 0:
        raise ValueError("unreachable_grace must not be negative")
    if stale_after < 0:
        raise ValueError("stale_after must not be negative")

    if list_measurements is None:
        from qimchi_connect.client import list_live_measurements_sync

        def list_measurements(ws_url: str, probe_timeout: float) -> list[str]:
            """
            Probe an endpoint with the default synchronous client.

            Args:
                ws_url (str): WebSocket endpoint URL.
                probe_timeout (float): Budget in seconds for the connection
                    and for the response, applied to each separately.

            Returns:
                list[str]: Dataset identifiers advertised by the endpoint.

            """
            return list_live_measurements_sync(
                ws_url, timeout=probe_timeout, connect_timeout=probe_timeout
            )

    now = datetime.datetime.now(datetime.timezone.utc)
    stale: list[str] = []
    by_endpoint: dict[str, list[LiveMeasurement]] = {}
    for record, liveness in _live_records(stale_after):
        if liveness == "beating":
            # The producer says it is running, so no probe is needed.
            _note_reachable(record.measurement_id)
        elif liveness == "stopped":
            # The beats stopped. Hiding the row from the live list is not
            # enough: it still reads live_status = 1, and cleanup only
            # deletes rows that were ended.
            end_measurement(record.measurement_id)
            stale.append(record.measurement_id)
        else:
            # No heartbeat to read: a row written through
            # register_measurement, a publication that opted out of one, or a
            # producer whose heartbeat thread died while it kept serving. Ask
            # the endpoint.
            by_endpoint.setdefault(record.ws_url, []).append(record)

    for ws_url, records in by_endpoint.items():
        advertised: set[str] | None = None
        for attempt in range(retries):
            try:
                advertised = set(list_measurements(ws_url, timeout))
                break
            except Exception as exc:
                if attempt == retries - 1:
                    logger.info(
                        "Live endpoint %s failed %d probe(s): %s",
                        ws_url,
                        retries,
                        exc,
                    )
        for record in records:
            if advertised is not None:
                # The endpoint answered, so it reports what it serves.
                # Absent means ended; present clears any earlier miss.
                if record.measurement_id in advertised:
                    _note_reachable(record.measurement_id)
                else:
                    end_measurement(record.measurement_id)
                    stale.append(record.measurement_id)
                continue

            unanswered_for = _note_unreachable(record.measurement_id, now)
            if unanswered_for >= unreachable_grace:
                end_measurement(record.measurement_id)
                stale.append(record.measurement_id)
    return tuple(stale)


def maintain_registry(
    retention_days: int = 7,
    *,
    timeout: float = 1.0,
    retries: int = 2,
    stale_after: float = DEFAULT_STALE_AFTER,
    unreachable_grace: float = DEFAULT_UNREACHABLE_GRACE,
    list_measurements: Callable[[str, float], list[str]] | None = None,
) -> RegistryMaintenanceResult:
    """
    Reconcile stale live rows and delete expired ended rows.

    Args:
        retention_days (int): Retention period for ended rows in whole days.
        timeout (float): Budget in seconds for each endpoint attempt, applied
            to the connection and to the response separately.
        retries (int): Number of endpoint attempts before declaring failure.
        stale_after (float): Seconds a producer's heartbeat may go unrefreshed
            before its measurements are ended.
        unreachable_grace (float): Seconds an endpoint may stay unanswerable
            before the measurements it carried are ended.
        list_measurements (Callable[[str, float], list[str]] | None): Optional probe
            used for testing or custom transports.

    Returns:
        RegistryMaintenanceResult: Stale identifiers and deletion count.

    Raises:
        ValueError: If a timeout, retry, grace, or retention argument is
            invalid.

    """
    stale = reconcile_live_measurements(
        timeout=timeout,
        retries=retries,
        stale_after=stale_after,
        unreachable_grace=unreachable_grace,
        list_measurements=list_measurements,
    )
    deleted = cleanup_old_measurements(days=retention_days)
    return RegistryMaintenanceResult(stale, deleted)
