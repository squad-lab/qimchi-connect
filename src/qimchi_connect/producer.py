"""
High-level lifecycle API for publishing live measurements.

"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Mapping

import xarray as xr

from qimchi_connect import registry, server
from qimchi_connect.server import SnapshotProvider

logger = logging.getLogger(__name__)

_REGISTRATIONS: dict[str, LiveMeasurementRegistration] = {}
_REGISTRATION_LOCK = threading.RLock()

# One beat covers every measurement this process publishes, so a process
# running several pays for one thread and one write per interval.
_HEARTBEAT_THREAD: threading.Thread | None = None
_HEARTBEAT_STOP = threading.Event()


def _heartbeat_loop(interval: float) -> None:
    """
    Stamp every open registration until asked to stop.

    Args:
        interval (float): Seconds between rounds.

    """
    while not _HEARTBEAT_STOP.wait(interval):
        with _REGISTRATION_LOCK:
            measurement_ids = list(_REGISTRATIONS)
        for measurement_id in measurement_ids:
            try:
                registry.heartbeat(measurement_id)
            except Exception:
                # A registry that cannot be written -- locked, read-only, on a
                # disconnected share -- must not take the measurement down. The
                # measurement drops out of the live list, which is what a
                # consumer would conclude from the missing beats anyway.
                logger.warning(
                    "Could not record a heartbeat for %s",
                    measurement_id,
                    exc_info=True,
                )


def _ensure_heartbeat(interval: float) -> None:
    """
    Start the shared heartbeat thread unless it is already running.

    The caller holds ``_REGISTRATION_LOCK``. The first registration in a
    process sets the cadence for the rest.

    Args:
        interval (float): Seconds between rounds. Zero or less publishes
            without a heartbeat, leaving the measurement to be probed.

    """
    global _HEARTBEAT_THREAD
    if interval <= 0:
        return
    if _HEARTBEAT_THREAD is not None and _HEARTBEAT_THREAD.is_alive():
        return

    _HEARTBEAT_STOP.clear()
    _HEARTBEAT_THREAD = threading.Thread(
        target=_heartbeat_loop,
        args=(interval,),
        daemon=True,
        name="qimchi-connect-heartbeat",
    )
    _HEARTBEAT_THREAD.start()


def stop_heartbeat() -> None:
    """
    Stop the shared heartbeat thread and wait briefly for it.

    Must not be called while holding ``_REGISTRATION_LOCK``: the thread takes
    that lock every round, so joining under it deadlocks.

    """
    global _HEARTBEAT_THREAD
    thread = _HEARTBEAT_THREAD
    _HEARTBEAT_STOP.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    _HEARTBEAT_THREAD = None


class QCoDeSSnapshotProvider:
    """
    Snapshot provider for a running QCoDeS measurement.

    A QCoDeS run needs two things a plain snapshot callback does not provide:

    1. **Thread affinity.** The run holds a SQLite connection usable only from
       the thread that opened it -- the measurement thread. A snapshot
       callback runs on the server's own thread whenever a client asks for
       data, where reading raises ``sqlite3.ProgrammingError``.
       ``dataset.cache`` behaves the same way, since it still consults the
       connection for run metadata. The measurement thread therefore refreshes
       a cached snapshot, and the server thread only reads that cache, under a
       lock.
    2. **Batched writes.** QCoDeS does not flush every ``add_result`` to disk,
       so even a same-thread read can miss the most recent points until the
       next flush. ``refresh`` forces a synchronous flush before reading.

    ``to_xarray_dataset`` re-reads the whole run each time, so refreshing on
    every point is quadratic in the number of points. ``refresh`` is throttled
    by ``refresh_interval``; calling it once per point is the intended usage.

    The datasaver is duck-typed, so ``qcodes`` is never imported here and
    stays an optional dependency.

    """

    #: Producer metadata applied when the caller does not pass its own.
    metadata: Mapping[str, Any] = {"source_package": "qcodes"}

    def __init__(self, datasaver: Any, *, refresh_interval: float = 0.25) -> None:
        """
        Wrap a QCoDeS datasaver as a live snapshot source.

        Args:
            datasaver (Any): The object yielded by ``Measurement.run()``, or
                anything exposing ``flush_data_to_database`` and a ``dataset``
                with ``to_xarray_dataset``.
            refresh_interval (float): Minimum seconds between real refreshes.
                Zero refreshes on every call.

        Raises:
            ValueError: If ``refresh_interval`` is negative.

        """
        if refresh_interval < 0:
            raise ValueError("refresh_interval must not be negative")

        self._datasaver = datasaver
        self._refresh_interval = float(refresh_interval)
        self._lock = threading.Lock()
        self._snapshot = xr.Dataset()
        self._last_refresh = float("-inf")

    def refresh(self, *, force: bool = False) -> None:
        """
        Re-read the run into the cached snapshot. Call from the measurement thread.

        Args:
            force (bool): Refresh even if ``refresh_interval`` has not elapsed.
                Used for the initial seed and the final snapshot.

        """
        now = time.monotonic()
        if not force and now - self._last_refresh < self._refresh_interval:
            return

        # Both calls touch the run's SQLite connection, so they have to happen
        # here, on the caller's (measurement) thread -- never in __call__.
        self._datasaver.flush_data_to_database(block=True)
        snapshot = self._datasaver.dataset.to_xarray_dataset()

        with self._lock:
            self._snapshot = snapshot
        self._last_refresh = now

    def add_result(self, *results: Any) -> None:
        """
        Record one point and refresh, so a measurement loop needs one call.

        Args:
            *results (Any): Passed straight to ``datasaver.add_result``.

        """
        self._datasaver.add_result(*results)
        self.refresh()

    def prepare(self) -> None:
        """
        Seed the cache before the measurement is published.

        Called by ``register_live_measurement`` so a client that connects
        immediately sees the run so far.

        """
        self.refresh(force=True)

    def __call__(self) -> xr.Dataset:
        """
        Return the cached snapshot. Safe to call from any thread.

        Returns:
            xr.Dataset: Most recent refreshed snapshot of the run.

        """
        with self._lock:
            return self._snapshot


class QanarySnapshotProvider:
    """
    Snapshot provider for a running qanary measurement.

    qanary keeps its live data in an in-memory Zarr store that the sweep
    writes to as it goes. This reads the current contents of that store on
    every call, so a client always sees the run so far.

    The store is opened and closed per call rather than held open, because the
    sweep is still writing to it and the read happens on the server's thread.

    """

    #: Producer metadata applied when the caller does not pass its own.
    metadata: Mapping[str, Any] = {
        "source_package": "qanary",
        "source_format": "zarr",
    }

    def __init__(self, store: Any) -> None:
        """
        Wrap an in-memory Zarr store as a live snapshot source.

        Args:
            store (Any): Zarr store the measurement writes into, typically a
                ``zarr.MemoryStore``.

        """
        self._store = store

    def __call__(self) -> xr.Dataset:
        """
        Read the current contents of the store. Safe to call from any thread.

        Returns:
            xr.Dataset: The measurement as written so far.

        """
        with xr.open_zarr(store=self._store, consolidated=False) as dataset:
            return dataset.load()


class QuantifySnapshotProvider:
    """
    Snapshot provider for a running Quantify measurement.

    Quantify writes the in-progress dataset to disk as it acquires, and
    ``load_dataset`` re-reads it, so unlike QCoDeS there is no thread affinity
    to work around. What this owns is the convention: deriving the measurement
    id and the on-disk location from a tuid, naming the producer, and covering
    the window at the start of a run where the file exists but is not yet
    readable.

    ``quantify_core`` is imported lazily, so the rest of the package stays
    usable without it.

    Example:
        >>> from quantify_core.data.handling import get_tuids_containing
        >>> tuid = get_tuids_containing("my-experiment")[-1]
        >>> provider = QuantifySnapshotProvider(tuid)
        >>> with live_measurement(provider.measurement_id, provider):
        ...     MC.run("my-experiment")

    """

    def __init__(self, tuid: str) -> None:
        """
        Wrap a Quantify experiment as a live snapshot source.

        Args:
            tuid (str): Quantify tuid of the run to publish.

        Raises:
            ImportError: If ``quantify_core`` is not installed.

        """
        try:
            from quantify_core.data.handling import (
                load_dataset,
                locate_experiment_container,
            )
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "QuantifySnapshotProvider needs quantify-core; install "
                "qimchi-connect[quantify]"
            ) from exc

        self._tuid = str(tuid)
        self._load_dataset = load_dataset
        self._locate_container = locate_experiment_container
        self._snapshot = xr.Dataset()

    @property
    def tuid(self) -> str:
        """
        Return the tuid of the published run.

        Returns:
            str: Quantify tuid.

        """
        return self._tuid

    @property
    def measurement_id(self) -> str:
        """
        Return the identifier to publish this run under.

        Returns:
            str: The tuid, prefixed so it is recognisable among measurements
                from other frameworks.

        """
        return f"quantify-{self._tuid}"

    @property
    def metadata(self) -> Mapping[str, Any]:
        """
        Return the producer metadata for this run.

        Returns:
            Mapping[str, Any]: Producer identification.

        """
        return {"source_package": "quantify", "source_format": "hdf5"}

    @property
    def disk_path(self) -> Path | None:
        """
        Return the dataset file Qimchi should fall back to.

        Returns:
            Path | None: The run's ``dataset.hdf5``, or ``None`` while
                Quantify has yet to create its experiment container.

        """
        try:
            return Path(self._locate_container(self._tuid)) / "dataset.hdf5"
        except Exception:
            return None

    def __call__(self) -> xr.Dataset:
        """
        Re-read the run from disk. Safe to call from any thread.

        A read that fails returns the previous snapshot rather than raising:
        Quantify rewrites the file as it acquires, so a client polling during a
        write would otherwise see an error instead of the run so far.

        Returns:
            xr.Dataset: The run as last read, empty before the first success.

        """
        try:
            self._snapshot = self._load_dataset(self._tuid)
        except Exception:
            logger.debug(
                "Quantify dataset for %s not readable yet; serving the "
                "previous snapshot",
                self._tuid,
                exc_info=True,
            )
        return self._snapshot


@dataclass(slots=True)
class LiveMeasurementRegistration:
    """
    Handle for one published live measurement.

    Returned by :func:`register_live_measurement` and :func:`live_measurement`. Use it
    to update the advertised disk path; closing it stops publication and marks
    the discovery record ended.

    """

    measurement_id: str
    ws_url: str
    ws_port: int
    disk_path: str | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def update_disk_path(self, disk_path: str | Path) -> None:
        """
        Update the persisted fallback path advertised for this measurement.

        Args:
            disk_path (str | Path): New persisted measurement location.

        Raises:
            RuntimeError: If the registration has already been closed.

        """
        if self._closed:
            raise RuntimeError("live measurement registration is closed")
        self.disk_path = str(disk_path)
        registry.update_measurement_path(self.measurement_id, disk_path)

    def close(self) -> None:
        """
        Stop advertising the measurement and mark its discovery record ended.

        """
        if self._closed:
            return

        with _REGISTRATION_LOCK:
            if _REGISTRATIONS.get(self.measurement_id) is self:
                _REGISTRATIONS.pop(self.measurement_id, None)
                server.unregister_snapshot_provider(self.measurement_id)
                registry.end_measurement(self.measurement_id)
            self._closed = True
            remaining = len(_REGISTRATIONS)

        # Outside the lock: stop_heartbeat joins a thread that takes it.
        if not remaining:
            stop_heartbeat()

    def __enter__(self) -> LiveMeasurementRegistration:
        """
        Return this active registration to a context manager.

        Returns:
            LiveMeasurementRegistration: This active registration.

        """
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close the registration when its context exits.

        Args:
            exception_type (type[BaseException] | None): Exception type from the
                managed block, if any.
            exception (BaseException | None): Exception from the managed block,
                if any.
            traceback (TracebackType | None): Associated traceback, if any.

        """
        self.close()


def register_live_measurement(
    measurement_id: str,
    snapshot: SnapshotProvider | QCoDeSSnapshotProvider,
    *,
    disk_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    host: str = "localhost",
    port: int = 0,
    maintain: bool = True,
    retention_days: int = 7,
    heartbeat_interval: float = registry.DEFAULT_HEARTBEAT_INTERVAL,
) -> LiveMeasurementRegistration:
    """
    Publish a live xarray snapshot callback and register it for discovery.

    Args:
        measurement_id (str): Stable identifier shown to live-data consumers.
        snapshot (SnapshotProvider | QCoDeSSnapshotProvider): Callback
            returning the current measurement, or a provider that also knows how
            to seed and describe itself (see ``QCoDeSSnapshotProvider``). The
            server takes ownership of each returned dataset, materializes it,
            and closes its backing resources.
        disk_path (str | Path | None): Optional persisted fallback location.
        metadata (Mapping[str, Any] | None): Optional producer metadata.
            Defaults to the provider's own ``metadata`` when it has one.
        host (str): WebSocket bind host used when starting a server.
        port (int): WebSocket port, or zero for an operating-system choice.
        maintain (bool): Whether to reconcile and prune the registry first.
        retention_days (int): Retention period for ended discovery rows.
        heartbeat_interval (float): Seconds between the stamps that tell
            consumers this process is still running. Zero or less publishes
            without one, leaving consumers to probe the endpoint instead. One
            thread beats for every measurement in the process, so the first
            publication sets the cadence and later ones inherit it.

    Returns:
        LiveMeasurementRegistration: Handle used to update or close publication.

    Raises:
        RuntimeError: If the WebSocket server cannot be started.
        TypeError: If the identifier or snapshot callback is invalid.
        ValueError: If an identifier or maintenance argument is invalid.

    """
    # A provider that knows its own framework describes itself: it can name
    # the producing package and seed its cache before anyone can query it.
    # A plain callback has neither, and needs neither.
    if metadata is None:
        metadata = getattr(snapshot, "metadata", None)
    prepare = getattr(snapshot, "prepare", None)
    if callable(prepare):
        prepare()

    with _REGISTRATION_LOCK:
        existing = _REGISTRATIONS.get(measurement_id)
        if existing is not None:
            existing.close()

        if maintain:
            # Housekeeping over rows other runs left behind. A registry that
            # cannot be swept -- locked, read-only, mid-migration -- must not
            # stop this measurement from being published.
            try:
                result = registry.maintain_registry(retention_days=retention_days)
                if result.stale_measurement_ids or result.deleted_count:
                    logger.info(
                        "Live registry maintenance marked %d stale and deleted "
                        "%d old measurement record(s)",
                        len(result.stale_measurement_ids),
                        result.deleted_count,
                    )
            except Exception:
                logger.warning(
                    "Could not maintain the live registry before publishing %s",
                    measurement_id,
                    exc_info=True,
                )

        server.register_snapshot_provider(measurement_id, snapshot, metadata)
        if not server.is_server_running() and not server.start_live_server(host, port):
            server.unregister_snapshot_provider(measurement_id)
            raise RuntimeError("failed to start the live WebSocket server")

        active_host = server.get_server_host() or host
        active_port = server.get_server_port()
        ws_url = f"ws://{active_host}:{active_port}"
        try:
            registry.register_measurement(
                measurement_id, disk_path, ws_url, active_port
            )
        except Exception:
            server.unregister_snapshot_provider(measurement_id)
            raise

        registration = LiveMeasurementRegistration(
            measurement_id=measurement_id,
            ws_url=ws_url,
            ws_port=active_port,
            disk_path=str(disk_path) if disk_path is not None else None,
        )
        _REGISTRATIONS[measurement_id] = registration
        if heartbeat_interval >= registry.DEFAULT_STALE_AFTER:
            logger.warning(
                "Heartbeat interval of %.1fs for %s is longer than the default "
                "staleness window of %.1fs, so consumers using that window will "
                "treat this measurement as dead between beats",
                heartbeat_interval,
                measurement_id,
                registry.DEFAULT_STALE_AFTER,
            )
        if heartbeat_interval > 0:
            # Stamp before returning, so the measurement never spends its first
            # interval looking like a producer too old to have a heartbeat.
            registry.heartbeat(measurement_id)
            _ensure_heartbeat(heartbeat_interval)
        return registration


def get_live_registration(measurement_id: str) -> LiveMeasurementRegistration | None:
    """
    Return the open publication for a measurement, if there is one.

    Args:
        measurement_id (str): Identifier to resolve.

    Returns:
        LiveMeasurementRegistration | None: The open registration, or ``None``
            when the measurement is not currently published.

    """
    with _REGISTRATION_LOCK:
        return _REGISTRATIONS.get(measurement_id)


def close_live_measurement(measurement_id: str) -> bool:
    """
    Stop publishing a measurement, addressed by its identifier.

    Lets a producer that publishes and closes in different places do so
    without carrying the handle between them.

    Args:
        measurement_id (str): Identifier to stop publishing.

    Returns:
        bool: Whether a publication was open and has now been closed.

    """
    registration = get_live_registration(measurement_id)
    if registration is None:
        return False
    registration.close()
    return True


def update_live_disk_path(measurement_id: str, disk_path: str | Path) -> bool:
    """
    Update the persisted path advertised for a published measurement.

    Args:
        measurement_id (str): Identifier to update.
        disk_path (str | Path): New persisted measurement location.

    Returns:
        bool: Whether a publication was open and has now been updated.

    """
    registration = get_live_registration(measurement_id)
    if registration is None:
        return False
    registration.update_disk_path(disk_path)
    return True


@contextmanager
def live_measurement(
    measurement_id: str,
    snapshot: SnapshotProvider | QCoDeSSnapshotProvider,
    **options: Any,
) -> Iterator[LiveMeasurementRegistration]:
    """
    Publish a measurement for the duration of a managed acquisition block.

    Args:
        measurement_id (str): Stable identifier shown to live-data consumers.
        snapshot (SnapshotProvider | QCoDeSSnapshotProvider): Callback
            returning the current measurement, or a self-describing provider.
        **options (Any): Options forwarded to ``register_live_measurement``.

    Yields:
        LiveMeasurementRegistration: Active publication handle.

    Raises:
        RuntimeError: If the WebSocket server cannot be started.
        TypeError: If the identifier or snapshot callback is invalid.
        ValueError: If an identifier or maintenance argument is invalid.

    """
    registration = register_live_measurement(measurement_id, snapshot, **options)
    try:
        yield registration
    finally:
        registration.close()
