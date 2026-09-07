"""
WebSocket client for loading package-neutral live measurements.

"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any

import websockets
import xarray as xr

from qimchi_connect.protocol import (
    MAX_MESSAGE_SIZE,
    PROTOCOL_VERSION,
    measurement_from_snapshot,
    unpack_snapshot,
)

DEFAULT_WS_URL = "ws://localhost:8765"

# Connections and handshakes are expected to be quick, but serializing
# a large measurement can take far longer. So, the two timeouts are separate.
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_TIMEOUT = 30.0


def _validate_since_rows(since_rows: int | None) -> int | None:
    """Validate and normalize an incremental-snapshot row offset."""
    if since_rows is None:
        return None
    if isinstance(since_rows, bool) or not isinstance(since_rows, int):
        raise TypeError("since_rows must be an integer or None")
    if since_rows < 0:
        raise ValueError("since_rows must not be negative")
    return since_rows


async def send_request(
    request: dict[str, Any],
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> dict[str, Any]:
    """
    Send one request over a short-lived WebSocket connection.

    Args:
        request (dict[str, Any]): JSON-compatible request object.
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds, measured from the moment
            the connection is open.
        connect_timeout (float): Connection and handshake timeout in seconds.

    Returns:
        dict[str, Any]: Decoded response object.

    Raises:
        RuntimeError: If the server returns a non-object JSON response.

    """
    websocket = await websockets.connect(
        ws_url,
        max_size=MAX_MESSAGE_SIZE,
        open_timeout=connect_timeout,
        # The server never offers permessage-deflate (see server.py's
        # websockets.serve call), so the connection runs uncompressed
        # either way, also making it faster.
        compression=None,
    )
    async with websocket:
        async with asyncio.timeout(timeout):
            await websocket.send(json.dumps(request))
            raw = await websocket.recv()

    # websockets hands back `bytes` for a binary frame and `str` for a text
    # one, matching what the server sent (see server._handle_client) -- only a
    # successful get_snapshot response is ever binary, so every other action,
    # and get_snapshot's own error responses, still take the plain-JSON path.
    if isinstance(raw, (bytes, bytearray)):
        response = unpack_snapshot(raw)
    else:
        response = json.loads(raw)
    if not isinstance(response, dict):
        raise RuntimeError("Live server returned a non-object response")
    return response


async def list_live_measurements(
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> list[str]:
    """
    List measurement identifiers advertised by a live server.

    Args:
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.

    Returns:
        list[str]: Advertised measurement identifiers.

    Raises:
        RuntimeError: If the server reports a request failure.

    """
    response = await send_request(
        {"action": "list"}, ws_url, timeout=timeout, connect_timeout=connect_timeout
    )
    if not response.get("success"):
        raise RuntimeError(f"Failed to list live measurements: {response.get('error')}")
    return response.get("measurements", [])


async def get_live_snapshot(
    measurement_id: str,
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    since_rows: int | None = None,
) -> dict[str, Any]:
    """
    Fetch one atomic measurement snapshot.

    Args:
        measurement_id (str): Identifier advertised by the server.
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.
        since_rows (int | None): Rows the caller already holds. The server
            answers with the rows beyond that instead of the whole
            measurement, which is what keeps a long sweep cheap to follow. A
            server that predates this field ignores it and sends everything,
            and the response then carries no ``rows_from``.

    Returns:
        dict[str, Any]: Snapshot response containing xarray structure and data.

    Raises:
        RuntimeError: If the server reports a request failure.
        TypeError: If ``since_rows`` is not an integer or None.
        ValueError: If ``since_rows`` is negative.

    """
    since_rows = _validate_since_rows(since_rows)
    request = {
        "action": "get_snapshot",
        "measurement_id": measurement_id,
        "protocol_version": PROTOCOL_VERSION,
    }
    if since_rows is not None:
        request["since_rows"] = since_rows
    response = await send_request(
        request, ws_url, timeout=timeout, connect_timeout=connect_timeout
    )
    if not response.get("success"):
        raise RuntimeError(
            f"Failed to get snapshot for {measurement_id}: {response.get('error')}"
        )
    return response


async def open_live_measurement(
    measurement_id: str,
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    since_rows: int | None = None,
) -> xr.Dataset:
    """
    Load a live measurement in one WebSocket round trip.

    Args:
        measurement_id (str): Identifier advertised by the server.
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.
        since_rows (int | None): Rows the caller already holds, so only the
            rows beyond them are fetched.

    Returns:
        xr.Dataset: The measurement, or the rows past ``since_rows`` when the
            server answered with those. When the server supports incremental
            snapshots, ``encoding["qimchi_connect_rows"]`` says where the
            returned rows belong -- ``rows_from``, ``rows_written``,
            ``rows_total`` and ``append_dim``. A full response then has
            ``rows_from == 0``. The encoding is absent for older servers and
            datasets that cannot be split row-wise.

    Raises:
        RuntimeError: If the server reports a request failure.
        TypeError: If ``since_rows`` is not an integer or None.
        ValueError: If ``since_rows`` is negative.

    """
    snapshot = await get_live_snapshot(
        measurement_id,
        ws_url,
        timeout=timeout,
        connect_timeout=connect_timeout,
        since_rows=since_rows,
    )
    dataset = measurement_from_snapshot(snapshot)
    source = snapshot.get("source")
    if isinstance(source, dict):
        dataset.encoding["qimchi_connect_source"] = source
    if "rows_from" in snapshot:
        dataset.encoding["qimchi_connect_rows"] = {
            key: snapshot[key]
            for key in ("append_dim", "rows_from", "rows_written", "rows_total")
            if key in snapshot
        }
    return dataset


async def get_measurement_info(
    measurement_id: str,
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> dict[str, Any]:
    """
    Fetch measurement structure and global attributes without array values.

    Args:
        measurement_id (str): Identifier advertised by the server.
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.

    Returns:
        dict[str, Any]: Coordinate names, data-variable names, and attributes.

    Raises:
        RuntimeError: If the server reports a request failure.

    """
    request = {
        "action": "get_data",
        "measurement_id": measurement_id,
        "protocol_version": PROTOCOL_VERSION,
    }
    response = await send_request(
        request, ws_url, timeout=timeout, connect_timeout=connect_timeout
    )
    if not response.get("success"):
        raise RuntimeError(f"Failed to get measurement info: {response.get('error')}")
    return {
        "coords": response.get("coords", []),
        "data_vars": response.get("data_vars", []),
        "attrs": response.get("attrs", {}),
    }


async def get_measurement_data(
    measurement_id: str,
    variables: list[str],
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> dict[str, Any]:
    """
    Fetch values for selected variables.

    Args:
        measurement_id (str): Identifier advertised by the server.
        variables (list[str]): Coordinate or data-variable names to retrieve.
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.

    Returns:
        dict[str, Any]: Mapping from available requested names to values.

    Raises:
        RuntimeError: If the server reports a request failure.

    """
    request = {
        "action": "get_data",
        "measurement_id": measurement_id,
        "protocol_version": PROTOCOL_VERSION,
        "variables": variables,
    }
    response = await send_request(
        request, ws_url, timeout=timeout, connect_timeout=connect_timeout
    )
    if not response.get("success"):
        raise RuntimeError(f"Failed to get measurement data: {response.get('error')}")
    return response.get("data", {})


def _run_sync(coroutine: Any, timeout: float) -> Any:
    """
    Run an asynchronous client operation from synchronous code.

    Args:
        coroutine (Any): Coroutine object to execute.
        timeout (float): Maximum number of seconds to wait.

    Returns:
        Any: Coroutine result.

    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coroutine)
        return future.result(timeout=timeout + 1.0)


def list_live_measurements_sync(
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> list[str]:
    """
    Synchronously list measurement identifiers advertised by a server.

    Args:
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.

    Returns:
        list[str]: Advertised measurement identifiers.

    Raises:
        RuntimeError: If the server reports a request failure.

    """
    return _run_sync(
        list_live_measurements(
            ws_url, timeout=timeout, connect_timeout=connect_timeout
        ),
        timeout + connect_timeout,
    )


def open_live_measurement_sync(
    measurement_id: str,
    ws_url: str = DEFAULT_WS_URL,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    since_rows: int | None = None,
) -> xr.Dataset:
    """
    Synchronously load a live measurement.

    Args:
        measurement_id (str): Identifier advertised by the server.
        ws_url (str): WebSocket endpoint URL.
        timeout (float): Response timeout in seconds.
        connect_timeout (float): Connection and handshake timeout in seconds.
        since_rows (int | None): Rows the caller already holds.

    Returns:
        xr.Dataset: As :func:`open_live_measurement`.

    Raises:
        RuntimeError: If the server reports a request failure.
        TypeError: If ``since_rows`` is not an integer or None.
        ValueError: If ``since_rows`` is negative.

    """
    return _run_sync(
        open_live_measurement(
            measurement_id,
            ws_url,
            timeout=timeout,
            connect_timeout=connect_timeout,
            since_rows=since_rows,
        ),
        timeout + connect_timeout,
    )
