"""
Background WebSocket server for package-neutral live measurements.

"""

from __future__ import annotations

import asyncio
import json
import logging
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
    json_compatible,
    json_default,
    pack_snapshot,
    snapshot_payload,
)

logger = logging.getLogger(__name__)

SnapshotProvider = Callable[[], xr.Dataset]

SNAPSHOT_CACHE_TTL = 0.25


@dataclass(frozen=True, slots=True)
class _ProviderEntry:
    """
    One registered snapshot callback and the producer metadata beside it.

    """

    snapshot: SnapshotProvider
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _SnapshotCacheEntry:
    """
    One measurement's serialized snapshot, reused until its TTL expires.

    """

    payload: dict[str, Any]
    # The binary-framed message body, built once and shared by every waiter
    # alongside ``payload``.
    blob: bytes
    metadata: Mapping[str, Any]
    created: float
    # Value of _PROVIDER_EPOCH when the snapshot was built. An entry from an
    # older epoch describes a provider that has since been replaced or removed.
    epoch: int


_PROVIDERS: dict[str, _ProviderEntry] = {}
_PROVIDERS_LOCK = threading.RLock()
# Incremented whenever any provider is registered or removed, so a snapshot
# build already in flight cannot write its result into the cache under a
# provider that no longer exists.
_PROVIDER_EPOCH = 0

_SERVER: Any | None = None
_SERVER_THREAD: threading.Thread | None = None
_SERVER_LOOP: asyncio.AbstractEventLoop | None = None
_SERVER_HOST = ""
_SERVER_PORT = 0
_SERVER_ERROR: BaseException | None = None
_SERVER_READY = threading.Event()

# Read and written from the server's event loop thread, and dropped from the
# registering thread through _invalidate_snapshot. Locks are stored with the
# loop they belong to, since a lock cannot be awaited from a different one and
# callers may drive _process_request on a loop of their own.
_SNAPSHOT_CACHE: dict[str, _SnapshotCacheEntry] = {}
_SNAPSHOT_LOCKS: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}


def _invalidate_snapshot(measurement_id: str) -> None:
    """
    Drop a measurement's cached snapshot and open a new provider epoch.

    Call with ``_PROVIDERS_LOCK`` held, whenever a provider is registered or
    removed, so the next request rebuilds rather than answering from the
    previous provider's data.

    Args:
        measurement_id (str): Identifier whose cached snapshot is no longer valid.

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
            measurement as an ``xarray.Dataset``.
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


def _measurement_ids() -> list[str]:
    """
    Return the identifiers of every published measurement.

    Returns:
        list[str]: Currently available live measurement identifiers.

    """
    with _PROVIDERS_LOCK:
        return list(_PROVIDERS)


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

    dataset = entry.snapshot()
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"Snapshot provider for {measurement_id!r} returned "
            f"{type(dataset).__name__}, expected xarray.Dataset"
        )
    return dataset, entry.metadata


def _build_snapshot(
    measurement_id: str,
) -> tuple[dict[str, Any], bytes, Mapping[str, Any]]:
    """
    Open a measurement, serialize it, and frame the binary message.

    All three steps are blocking, so this is meant to be called off the event
    loop. Payload and blob are built from a single open dataset, so they
    always describe the same state.

    Args:
        measurement_id (str): Identifier to resolve.

    Returns:
        tuple[dict[str, Any], bytes, Mapping[str, Any]]: Snapshot payload, its
            packed binary message, and producer metadata.

    """
    dataset, metadata = _open_measurement(measurement_id)
    payload = snapshot_payload(dataset, binary=True)

    # The blob's JSON header is a full response, not just the snapshot
    # fields: the client reads success, the identifier and `source` out of
    # the unpacked header. The shape does not depend on the request, so it
    # is built once and cached with the rest.
    header = dict(payload)
    header.update(
        success=True,
        measurement_id=measurement_id,
        protocol=PROTOCOL_NAME,
        protocol_version=PROTOCOL_VERSION,
        source=json_compatible(dict(metadata)),
    )
    return payload, pack_snapshot(header, dataset), metadata


async def _cached_snapshot(
    measurement_id: str,
) -> tuple[dict[str, Any], bytes, Mapping[str, Any]]:
    """
    Return a recent snapshot, building at most one per measurement at a time.

    Requests that arrive while a build is in flight wait for it and share its
    result, so concurrent pollers of one measurement cost a single serialization
    per cache cycle. A result whose provider changed while it was being built
    is returned to the caller but not cached.

    Args:
        measurement_id (str): Identifier to resolve.

    Returns:
        tuple[dict[str, Any], bytes, Mapping[str, Any]]: Snapshot payload, its
            packed binary message, and producer metadata. Both are shared
            between callers and must not be mutated in place.

    """
    loop = asyncio.get_running_loop()
    bound = _SNAPSHOT_LOCKS.get(measurement_id)
    if bound is None or bound[0] is not loop:
        bound = (loop, asyncio.Lock())
        _SNAPSHOT_LOCKS[measurement_id] = bound

    async with bound[1]:
        epoch = _PROVIDER_EPOCH
        entry = _SNAPSHOT_CACHE.get(measurement_id)
        if (
            entry is not None
            and entry.epoch == epoch
            and time.monotonic() - entry.created < SNAPSHOT_CACHE_TTL
        ):
            return entry.payload, entry.blob, entry.metadata

        payload, blob, metadata = await asyncio.to_thread(
            _build_snapshot, measurement_id
        )
        if _PROVIDER_EPOCH == epoch:
            _SNAPSHOT_CACHE[measurement_id] = _SnapshotCacheEntry(
                payload=payload,
                blob=blob,
                metadata=metadata,
                created=time.monotonic(),
                epoch=epoch,
            )
        return payload, blob, metadata


def _build_data_payload(
    measurement_id: str, variables: list[str]
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """
    Build a ``get_data`` payload off the event loop.

    Args:
        measurement_id (str): Identifier to resolve.
        variables (list[str]): Names to retrieve, or empty for structure only.

    Returns:
        tuple[dict[str, Any], Mapping[str, Any]]: Response fields and producer
            metadata.

    """
    dataset, metadata = _open_measurement(measurement_id)
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


def _build_array_payload(measurement_id: str, array_path: str) -> dict[str, Any] | None:
    """
    Build a ``get_zarr_array`` payload off the event loop.

    Args:
        measurement_id (str): Identifier to resolve.
        array_path (str): Variable name to retrieve.

    Returns:
        dict[str, Any] | None: Response fields, or ``None`` when the variable is
            not present.

    """
    dataset, _metadata = _open_measurement(measurement_id)
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

    Locks are bound to the event loop that created them, so they must not
    outlive a server restart.

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
            # get_snapshot is answered with a single binary frame (see
            # protocol.pack_snapshot). Every other action, and get_snapshot
            # error responses, answer with JSON text.
            "snapshot_format": "xarray-binary",
        }

    if action == "list":
        return _response(success=True, measurements=_measurement_ids())

    if not measurement_id:
        return _response(success=False, error="measurement_id required")

    if _provider_entry(measurement_id) is None:
        return _response(
            success=False,
            error=f"Measurement {measurement_id} not found",
        )

    try:
        if action == "get_snapshot":
            payload, blob, metadata = await _cached_snapshot(measurement_id)
            response = _response(measurement_id, success=True, **payload)
            response["source"] = json_compatible(dict(metadata))
            # _handle_client pops this and sends it as one framed binary
            # message instead of JSON-encoding the response; see
            # protocol.pack_snapshot. It covers exactly the variables named
            # in payload["binary_vars"].
            response["_binary_payload"] = blob
            return response

        if action == "get_data":
            variables = request.get("variables", [])
            payload, metadata = await asyncio.to_thread(
                _build_data_payload, measurement_id, variables
            )
            response = _response(measurement_id, success=True, **payload)
            if not variables:
                response["source"] = json_compatible(dict(metadata))
            return response

        if action == "get_zarr_array":
            array_path = request.get("array_path", "")
            payload = await asyncio.to_thread(
                _build_array_payload, measurement_id, array_path
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
                # protocol.pack_snapshot uses
                # [header length][JSON header][concatenated arrays]. Sending
                # `bytes` makes websockets emit a binary frame, which is how
                # the client picks its decode path.
                await websocket.send(binary_payload)
            else:
                # Responses are already JSON-compatible, so they are encoded
                # directly; ``json_default`` covers anything a provider left
                # unconverted.
                await websocket.send(json.dumps(response, default=json_default))
    except websockets.ConnectionClosed:
        logger.debug("Live WebSocket client disconnected")


def _reserve_port(host: str) -> int:
    """
    Choose one free port up front so every address family binds the same one.

    A host such as ``localhost`` resolves to both ``127.0.0.1`` and ``::1``,
    and port zero would give each of them a different ephemeral port while
    only one of them can be advertised.

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
        """
        Create the WebSocket listener and wait until it closes.

        """
        global _SERVER, _SERVER_HOST, _SERVER_PORT
        bind_port = _reserve_port(host) if port == 0 else port
        _SERVER = await websockets.serve(
            _handle_client,
            host,
            bind_port,
            max_size=MAX_MESSAGE_SIZE,
            # permessage-deflate is on by default. It helps the small JSON
            # control messages (list, capabilities, errors) but actively hurts
            # snapshot payloads: raw float64 arrays are already near-maximum
            # entropy, so keeping compression off.
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
