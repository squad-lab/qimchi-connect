"""Background WebSocket server for live measurements."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import websockets
import xarray as xr
from websockets.asyncio.server import ServerConnection

from qimchi_connect.protocol import (
    MAX_MESSAGE_SIZE,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    append_dim,
    json_compatible,
    json_default,
    pack_snapshot,
    row_frontier,
    snapshot_payload,
)

logger = logging.getLogger(__name__)

SnapshotProvider = Callable[[], xr.Dataset]

DEFAULT_SNAPSHOT_CACHE_TTL = 0.25


def _configured_snapshot_cache_ttl() -> float:
    """Read a non-negative snapshot-cache lifetime from the environment."""
    raw = os.environ.get("QIMCHI_CONNECT_SNAPSHOT_TTL")
    if raw is None:
        return DEFAULT_SNAPSHOT_CACHE_TTL
    try:
        ttl = float(raw)
    except ValueError:
        ttl = math.nan
    if not math.isfinite(ttl) or ttl < 0:
        logger.warning(
            "Ignoring invalid QIMCHI_CONNECT_SNAPSHOT_TTL=%r; using %.2f seconds",
            raw,
            DEFAULT_SNAPSHOT_CACHE_TTL,
        )
        return DEFAULT_SNAPSHOT_CACHE_TTL
    return ttl


# Requests within this window reuse the same serialized snapshot. A longer TTL
# reduces serialization work at the cost of older data.
SNAPSHOT_CACHE_TTL = _configured_snapshot_cache_ttl()


@dataclass(frozen=True, slots=True)
class _ProviderEntry:
    """Registered snapshot callback and its producer metadata."""

    snapshot: SnapshotProvider
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _SnapshotCacheEntry:
    """Cached snapshot for one measurement."""

    payload: dict[str, Any]
    # Framed form of ``payload``, shared by all waiting requests.
    blob: bytes
    # Retained for incremental requests, which slice it without another read.
    dataset: xr.Dataset
    metadata: Mapping[str, Any]
    created: float
    # Provider epoch at build time.
    epoch: int


_PROVIDERS: dict[str, _ProviderEntry] = {}
_PROVIDERS_LOCK = threading.RLock()
# Incremented on provider changes to reject stale in-flight cache writes.
_PROVIDER_EPOCH = 0

_SERVER: Any | None = None
_SERVER_THREAD: threading.Thread | None = None
_SERVER_LOOP: asyncio.AbstractEventLoop | None = None
_SERVER_HOST = ""
_SERVER_PORT = 0
_SERVER_ERROR: BaseException | None = None
_SERVER_READY = threading.Event()

# Cache entries normally belong to the server loop but can be invalidated from
# a producer thread. Store each asyncio lock with its loop because tests and
# callers may run _process_request on a different loop.
_SNAPSHOT_CACHE: dict[str, _SnapshotCacheEntry] = {}
_SNAPSHOT_LOCKS: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}


def _invalidate_snapshot(measurement_id: str) -> None:
    """
    Invalidate a measurement's cache and advance the provider epoch.

    Call while holding ``_PROVIDERS_LOCK`` after registering or removing a
    provider.

    Args:
        measurement_id (str): Identifier to invalidate.

    """
    global _PROVIDER_EPOCH
    _PROVIDER_EPOCH += 1
    _SNAPSHOT_CACHE.pop(measurement_id, None)
    _SNAPSHOT_LOCKS.pop(measurement_id, None)


def register_snapshot_provider(
    measurement_id: str,
    snapshot: SnapshotProvider,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Register or replace a live measurement snapshot callback.

    Args:
        measurement_id (str): Stable identifier advertised to live clients.
        snapshot (SnapshotProvider): Callback returning the current
            ``xarray.Dataset``. The server loads and closes it, so the callback
            must transfer ownership of the returned object.
        metadata (Mapping[str, Any] | None): Optional producer metadata
            included in snapshot responses as ``source``.

    Raises:
        TypeError: If the identifier is not a string or the snapshot is not
            callable.
        ValueError: If the identifier is empty.

    """
    if not isinstance(measurement_id, str):
        raise TypeError("measurement_id must be a string")
    if not measurement_id.strip():
        raise ValueError("measurement_id must not be empty")
    if not callable(snapshot):
        raise TypeError("snapshot must be callable")

    with _PROVIDERS_LOCK:
        _PROVIDERS[measurement_id] = _ProviderEntry(snapshot, dict(metadata or {}))
        _invalidate_snapshot(measurement_id)


def unregister_snapshot_provider(measurement_id: str) -> None:
    """
    Remove a live snapshot callback if it is registered.

    Args:
        measurement_id (str): Identifier to remove.

    """
    with _PROVIDERS_LOCK:
        _PROVIDERS.pop(measurement_id, None)
        _invalidate_snapshot(measurement_id)


def _provider_entry(measurement_id: str) -> _ProviderEntry | None:
    """
    Return a registered snapshot provider.

    Args:
        measurement_id (str): Identifier to resolve.

    Returns:
        _ProviderEntry | None: Provider entry when registered, otherwise
            ``None``.

    """
    with _PROVIDERS_LOCK:
        return _PROVIDERS.get(measurement_id)


def _provider_state(measurement_id: str) -> tuple[_ProviderEntry, int] | None:
    """Return the provider and current epoch for a new request."""
    with _PROVIDERS_LOCK:
        entry = _PROVIDERS.get(measurement_id)
        if entry is None:
            return None
        return entry, _PROVIDER_EPOCH


def _measurement_ids() -> list[str]:
    """
    Return the identifiers of every published measurement.

    Returns:
        list[str]: Currently available live measurement identifiers.

    """
    with _PROVIDERS_LOCK:
        return list(_PROVIDERS)


def _open_provider(
    measurement_id: str, entry: _ProviderEntry
) -> tuple[xr.Dataset, Mapping[str, Any]]:
    """Call the provider retained when a request was accepted."""
    dataset = entry.snapshot()
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"Snapshot provider for {measurement_id!r} returned "
            f"{type(dataset).__name__}, expected xarray.Dataset"
        )
    return dataset, entry.metadata


def _open_measurement(measurement_id: str) -> tuple[xr.Dataset, Mapping[str, Any]]:
    """
    Call a measurement's snapshot provider.

    Args:
        measurement_id (str): Identifier to resolve.

    Returns:
        tuple[xr.Dataset, Mapping[str, Any]]: Current measurement and its
            producer metadata.

    Raises:
        KeyError: If the identifier is not registered.
        TypeError: If a provider returns a value other than ``xr.Dataset``.

    """
    entry = _provider_entry(measurement_id)
    if entry is None:
        raise KeyError(measurement_id)
    return _open_provider(measurement_id, entry)


def _row_fields(dataset: xr.Dataset, *, rows_from: int) -> dict[str, Any]:
    """
    Build incremental-position fields for a snapshot.

    Datasets without a shared leading dimension return no row fields and must
    be transferred in full.

    Args:
        dataset (xr.Dataset): Dataset being sent.
        rows_from (int): Index of its first row in the full measurement.

    Returns:
        dict[str, Any]: Row fields for the snapshot payload.

    """
    dim = append_dim(dataset)
    if dim is None:
        return {}
    return {
        "append_dim": dim,
        "rows_from": rows_from,
        "rows_written": rows_from + row_frontier(dataset, dim),
        "rows_total": rows_from + int(dataset.sizes.get(dim, 0)),
    }


def _partial_snapshot(
    measurement_id: str,
    dataset: xr.Dataset,
    metadata: Mapping[str, Any],
    since_rows: int,
) -> tuple[dict[str, Any], bytes] | None:
    """
    Serialize rows after the client's current offset.

    Args:
        measurement_id (str): Identifier the response is for.
        dataset (xr.Dataset): Whole current snapshot to slice.
        metadata (Mapping[str, Any]): Producer metadata for the header.
        since_rows (int): Number of rows the client already holds.

    Returns:
        tuple[dict[str, Any], bytes] | None: Partial payload and framed message,
            or ``None`` if the dataset cannot be sliced or the offset exceeds
            the written data.

    """
    dim = append_dim(dataset)
    if dim is None:
        return None

    rows_total = int(dataset.sizes.get(dim, 0))
    rows_written = row_frontier(dataset, dim)
    if since_rows > rows_written or since_rows > rows_total:
        return None

    sliced = dataset.isel({dim: slice(since_rows, rows_written)})
    payload = snapshot_payload(sliced, binary=True)
    payload.update(
        append_dim=dim,
        rows_from=since_rows,
        rows_written=rows_written,
        rows_total=rows_total,
    )
    header = dict(payload)
    header.update(
        success=True,
        measurement_id=measurement_id,
        protocol=PROTOCOL_NAME,
        protocol_version=PROTOCOL_VERSION,
        source=json_compatible(dict(metadata)),
    )
    return payload, pack_snapshot(header, sliced)


def _request_since_rows(request: Mapping[str, Any]) -> int | None:
    """Return a validated incremental-snapshot offset from a request."""
    since_rows = request.get("since_rows")
    if since_rows is None:
        return None
    if isinstance(since_rows, bool) or not isinstance(since_rows, int):
        raise TypeError("since_rows must be an integer or null")
    if since_rows < 0:
        raise ValueError("since_rows must not be negative")
    return since_rows


def _build_snapshot(
    measurement_id: str,
    provider: _ProviderEntry,
) -> tuple[dict[str, Any], bytes, xr.Dataset, Mapping[str, Any]]:
    """
    Open a measurement, serialize it, and frame the binary message.

    This blocking function runs outside the event loop. The payload and binary
    frame come from the same dataset instance.

    Args:
        measurement_id (str): Identifier to include in the response.
        provider (_ProviderEntry): Provider retained when the request was
            accepted.

    Returns:
        tuple[dict[str, Any], bytes, xr.Dataset, Mapping[str, Any]]: Snapshot
            payload, its packed binary message, the dataset both describe, and
            producer metadata.

    """
    dataset, metadata = _open_provider(measurement_id, provider)
    try:
        # Materialize lazy arrays before closing their backing files.
        dataset.load()
    finally:
        dataset.close()

    payload = snapshot_payload(dataset, binary=True)
    payload.update(_row_fields(dataset, rows_from=0))

    # The binary frame includes the complete response header and can therefore
    # be cached as-is.
    header = dict(payload)
    header.update(
        success=True,
        measurement_id=measurement_id,
        protocol=PROTOCOL_NAME,
        protocol_version=PROTOCOL_VERSION,
        source=json_compatible(dict(metadata)),
    )
    return payload, pack_snapshot(header, dataset), dataset, metadata


async def _cached_snapshot(
    measurement_id: str,
    provider: _ProviderEntry,
    provider_epoch: int,
) -> tuple[dict[str, Any], bytes, xr.Dataset, Mapping[str, Any]]:
    """
    Return a recent snapshot, building at most one per measurement at a time.

    Concurrent requests share one build. If the provider changes during that
    build, the result is returned to its original caller but not cached.

    Args:
        measurement_id (str): Identifier to resolve.
        provider (_ProviderEntry): Provider retained when the request was
            admitted.
        provider_epoch (int): Provider epoch at admission time.

    Returns:
        tuple[dict[str, Any], bytes, xr.Dataset, Mapping[str, Any]]: Snapshot
            payload, packed message, dataset, and producer metadata. Callers
            must not mutate these shared objects.

    """
    loop = asyncio.get_running_loop()
    bound = _SNAPSHOT_LOCKS.get(measurement_id)
    if bound is None or bound[0] is not loop:
        bound = (loop, asyncio.Lock())
        _SNAPSHOT_LOCKS[measurement_id] = bound

    async with bound[1]:
        entry = _SNAPSHOT_CACHE.get(measurement_id)
        if (
            entry is not None
            and entry.epoch == provider_epoch
            and time.monotonic() - entry.created < SNAPSHOT_CACHE_TTL
        ):
            return entry.payload, entry.blob, entry.dataset, entry.metadata

        payload, blob, dataset, metadata = await asyncio.to_thread(
            _build_snapshot, measurement_id, provider
        )
        # Keep provider validation and the cache write atomic with respect to
        # registration changes on producer threads.
        with _PROVIDERS_LOCK:
            if (
                _PROVIDER_EPOCH == provider_epoch
                and _PROVIDERS.get(measurement_id) is provider
            ):
                _SNAPSHOT_CACHE[measurement_id] = _SnapshotCacheEntry(
                    payload=payload,
                    blob=blob,
                    dataset=dataset,
                    metadata=metadata,
                    created=time.monotonic(),
                    epoch=provider_epoch,
                )
        return payload, blob, dataset, metadata


def _build_data_payload(
    measurement_id: str, provider: _ProviderEntry, variables: list[str]
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """
    Build a ``get_data`` payload off the event loop.

    Args:
        measurement_id (str): Identifier to resolve.
        provider (_ProviderEntry): Provider retained when the request was
            admitted.
        variables (list[str]): Names to retrieve, or empty for structure only.

    Returns:
        tuple[dict[str, Any], Mapping[str, Any]]: Response fields and producer
            metadata.

    """
    dataset, metadata = _open_provider(measurement_id, provider)
    attrs = json_compatible(dict(dataset.attrs))
    if not variables:
        return {
            "coords": [str(name) for name in dataset.coords],
            "data_vars": [str(name) for name in dataset.data_vars],
            "attrs": attrs,
        }, metadata
    data = {
        str(name): json_compatible(dataset[name].values)
        for name in variables
        if name in dataset.variables
    }
    return {"data": data, "attrs": attrs}, metadata


def _build_array_payload(
    measurement_id: str, provider: _ProviderEntry, array_path: str
) -> dict[str, Any] | None:
    """
    Build a ``get_zarr_array`` payload off the event loop.

    Args:
        measurement_id (str): Identifier to resolve.
        provider (_ProviderEntry): Provider retained when the request was
            admitted.
        array_path (str): Variable name to retrieve.

    Returns:
        dict[str, Any] | None: Response fields, or ``None`` when the variable is
            not present.

    """
    dataset, _metadata = _open_provider(measurement_id, provider)
    if not array_path or array_path not in dataset.variables:
        return None
    array = dataset[array_path]
    return {
        "array_path": array_path,
        "data": json_compatible(array.values),
        "shape": [int(size) for size in array.shape],
        "dtype": str(array.dtype),
        "dims": [str(dim) for dim in array.dims],
        "attrs": json_compatible(dict(array.attrs)),
    }


def _clear_snapshot_cache() -> None:
    """
    Drop cached snapshots and their per-measurement locks.

    Asyncio locks are bound to their event loop and cannot survive a restart.

    """
    _SNAPSHOT_CACHE.clear()
    _SNAPSHOT_LOCKS.clear()


def _request_measurement_id(request: Mapping[str, Any]) -> str | None:
    """
    Read the identifier field from a request.

    Args:
        request (Mapping[str, Any]): Decoded request object.

    Returns:
        str | None: Requested identifier when supplied.

    """
    return request.get("measurement_id")


def _response(measurement_id: str | None = None, **payload: Any) -> dict[str, Any]:
    """
    Build a response carrying the protocol identification fields.

    Args:
        measurement_id (str | None): Identifier associated with the response.
        **payload (Any): Response-specific fields.

    Returns:
        dict[str, Any]: Complete response object.

    """
    response = dict(payload)
    if measurement_id is not None:
        response["measurement_id"] = measurement_id
    response["protocol"] = PROTOCOL_NAME
    response["protocol_version"] = PROTOCOL_VERSION
    return response


async def _process_request(request: dict[str, Any]) -> dict[str, Any]:
    """
    Process one decoded WebSocket request.

    Args:
        request (dict[str, Any]): Request containing an action and optional
            measurement identifier.

    Returns:
        dict[str, Any]: JSON-compatible response object.

    """
    action = request.get("action")
    measurement_id = _request_measurement_id(request)

    if action == "capabilities":
        return {
            "success": True,
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "actions": [
                "capabilities",
                "list",
                "get_data",
                "get_snapshot",
                "get_zarr_array",
            ],
            "id_field": "measurement_id",
            # Successful snapshots use one binary frame; other responses use
            # JSON text.
            "snapshot_format": "xarray-binary",
        }

    if action == "list":
        return _response(success=True, measurements=_measurement_ids())

    if not measurement_id:
        return _response(success=False, error="measurement_id required")

    provider_state = _provider_state(measurement_id)
    if provider_state is None:
        return _response(
            success=False,
            error=f"Measurement {measurement_id} not found",
        )
    provider, provider_epoch = provider_state

    try:
        if action == "get_snapshot":
            since_rows = _request_since_rows(request)
            payload, blob, dataset, metadata = await _cached_snapshot(
                measurement_id, provider, provider_epoch
            )
            if since_rows is not None and since_rows > 0:
                partial = await asyncio.to_thread(
                    _partial_snapshot, measurement_id, dataset, metadata, since_rows
                )
                if partial is not None:
                    payload, blob = partial
            response = _response(measurement_id, success=True, **payload)
            response["source"] = json_compatible(dict(metadata))
            # _handle_client sends this frame instead of JSON-encoding the
            # response.
            response["_binary_payload"] = blob
            return response

        if action == "get_data":
            variables = request.get("variables", [])
            payload, metadata = await asyncio.to_thread(
                _build_data_payload, measurement_id, provider, variables
            )
            response = _response(measurement_id, success=True, **payload)
            if not variables:
                response["source"] = json_compatible(dict(metadata))
            return response

        if action == "get_zarr_array":
            array_path = request.get("array_path", "")
            payload = await asyncio.to_thread(
                _build_array_payload, measurement_id, provider, array_path
            )
            if payload is None:
                return _response(
                    success=False,
                    error=f"Variable {array_path!r} not found",
                )
            return _response(measurement_id, success=True, **payload)

        return _response(success=False, error=f"Unknown action: {action}")
    except Exception as exc:
        logger.exception("Failed live request for %s", measurement_id)
        return _response(success=False, error=str(exc))


async def _handle_client(websocket: ServerConnection) -> None:
    """
    Handle requests from one WebSocket connection.

    Args:
        websocket (ServerConnection): Connected WebSocket client.

    """
    try:
        async for message in websocket:
            try:
                request = json.loads(message)
                response = await _process_request(request)
            except json.JSONDecodeError:
                response = {"success": False, "error": "Invalid JSON"}
            except Exception as exc:
                logger.exception("Error processing WebSocket request")
                response = {"success": False, "error": str(exc)}
            binary_payload = response.pop("_binary_payload", None)
            if binary_payload is not None:
                # bytes select the client's binary snapshot decoder.
                await websocket.send(binary_payload)
            else:
                # json_default handles any provider values not converted yet.
                await websocket.send(json.dumps(response, default=json_default))
    except websockets.ConnectionClosed:
        logger.debug("Live WebSocket client disconnected")


def _reserve_port(host: str) -> int:
    """
    Reserve one port for every address family resolved from ``host``.

    With port zero, IPv4 and IPv6 listeners could otherwise receive different
    ephemeral ports.

    Args:
        host (str): Interface on which the server will listen.

    Returns:
        int: A port that was free at the time of the call.

    """
    family, socket_type, protocol, _canonical, address = socket.getaddrinfo(
        host or None, 0, type=socket.SOCK_STREAM
    )[0]
    with socket.socket(family, socket_type, protocol) as probe:
        probe.bind(address)
        return int(probe.getsockname()[1])


def _run_server(host: str, port: int) -> None:
    """
    Run the asynchronous server on its dedicated thread.

    Args:
        host (str): Interface on which to listen.
        port (int): Requested TCP port, or zero for an operating-system choice.

    """
    global _SERVER, _SERVER_ERROR, _SERVER_HOST, _SERVER_LOOP, _SERVER_PORT
    loop = asyncio.new_event_loop()
    _SERVER_LOOP = loop
    asyncio.set_event_loop(loop)

    async def serve() -> None:
        """Create the WebSocket listener and wait for it to close."""
        global _SERVER, _SERVER_HOST, _SERVER_PORT
        bind_port = _reserve_port(host) if port == 0 else port
        _SERVER = await websockets.serve(
            _handle_client,
            host,
            bind_port,
            max_size=MAX_MESSAGE_SIZE,
            # Numeric snapshot data rarely compresses enough to justify
            # permessage-deflate overhead.
            compression=None,
        )
        sockets = getattr(_SERVER, "sockets", None) or []
        _SERVER_HOST = host
        _SERVER_PORT = int(sockets[0].getsockname()[1]) if sockets else bind_port
        _SERVER_READY.set()
        await _SERVER.wait_closed()

    try:
        loop.run_until_complete(serve())
    except BaseException as exc:
        _SERVER_ERROR = exc
        _SERVER_READY.set()
        logger.exception("Live WebSocket server failed")
    finally:
        loop.close()


def start_live_server(host: str = "localhost", port: int = 8765) -> bool:
    """
    Start the process-wide live WebSocket server.

    Args:
        host (str): Interface on which to listen.
        port (int): Requested TCP port, or zero for an operating-system choice.

    Returns:
        bool: Whether the server became ready within the startup timeout.

    """
    global _SERVER_ERROR, _SERVER_THREAD
    if is_server_running():
        return True

    _SERVER_ERROR = None
    _SERVER_READY.clear()
    _clear_snapshot_cache()
    _SERVER_THREAD = threading.Thread(
        target=_run_server,
        args=(host, port),
        daemon=True,
        name="qimchi-connect-server",
    )
    _SERVER_THREAD.start()
    ready = _SERVER_READY.wait(timeout=2.0)
    return ready and _SERVER_ERROR is None and is_server_running()


def stop_live_server() -> None:
    """
    Stop the process-wide server and wait briefly for its thread.

    """
    global _SERVER, _SERVER_HOST, _SERVER_LOOP, _SERVER_PORT, _SERVER_THREAD
    server = _SERVER
    loop = _SERVER_LOOP
    thread = _SERVER_THREAD
    if server is not None and loop is not None and loop.is_running():
        loop.call_soon_threadsafe(server.close)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    _SERVER = None
    _SERVER_HOST = ""
    _SERVER_LOOP = None
    _SERVER_PORT = 0
    _SERVER_THREAD = None
    _clear_snapshot_cache()


def is_server_running() -> bool:
    """
    Check whether the process-wide server is ready.

    Returns:
        bool: Whether the server thread is alive and startup succeeded.

    """
    return (
        _SERVER_THREAD is not None
        and _SERVER_THREAD.is_alive()
        and _SERVER_READY.is_set()
        and _SERVER_ERROR is None
    )


def get_server_port() -> int:
    """
    Return the active server port.

    Returns:
        int: Active TCP port, or zero when stopped.

    """
    return _SERVER_PORT


def get_server_host() -> str:
    """
    Return the active server host.

    Returns:
        str: Bound host, or an empty string when stopped.

    """
    return _SERVER_HOST
