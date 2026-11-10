"""Serialization for the Qimchi live-data protocol."""

from __future__ import annotations

import base64
import datetime
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

PROTOCOL_NAME = "qimchi-connect"
PROTOCOL_VERSION = 1
# Leave room for JSON metadata alongside a large snapshot.
MAX_MESSAGE_SIZE = 1000 * 1024 * 1024

# A binary snapshot contains a 4-byte header length, a JSON header, and the raw
# arrays in ``binary_vars`` order. The entire snapshot fits in one WebSocket
# binary frame.
_HEADER_LENGTH = struct.Struct(">I")

_BYTES_MARKER = "__qimchi_connect_bytes__"
_COMPLEX_MARKER = "__qimchi_connect_complex__"


# These dtypes produce JSON-compatible scalars directly from ``tolist()``.
_PLAIN_ARRAY_KINDS = frozenset("biuf")


def json_default(value: Any) -> Any:
    """
    Convert one value that a JSON encoder could not serialize on its own.

    Use this as the ``default`` hook for :func:`json.dumps`.

    Args:
        value (Any): Value rejected by the JSON encoder.

    Returns:
        Any: A JSON-compatible representation of the value.

    Raises:
        TypeError: If the value has no supported representation.

    """
    if isinstance(value, (np.ndarray, np.generic, complex, bytes)):
        return json_compatible(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return json_compatible(list(value))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_compatible(value: Any) -> Any:
    """
    Convert a value into structures accepted by JSON encoders.

    Args:
        value (Any): Value to convert.

    Returns:
        Any: A JSON-compatible representation of the value.

    """
    if isinstance(value, np.ndarray):
        # Numeric and boolean arrays convert to plain Python scalars in one C-level
        # call; recursing per element only matters for object/complex/datetime data.
        if value.dtype.kind in _PLAIN_ARRAY_KINDS:
            return value.tolist()
        return json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, complex):
        return {_COMPLEX_MARKER: [value.real, value.imag]}
    if isinstance(value, bytes):
        return {_BYTES_MARKER: base64.b64encode(value).decode("ascii")}
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_compatible(item) for item in value]
    return value


def decode_json_value(value: Any) -> Any:
    """
    Restore values encoded with protocol-specific JSON markers.

    Args:
        value (Any): Decoded JSON value.

    Returns:
        Any: Value with supported marker objects restored.

    """
    if isinstance(value, dict):
        if set(value) == {_COMPLEX_MARKER}:
            real, imaginary = value[_COMPLEX_MARKER]
            return complex(real, imaginary)
        if set(value) == {_BYTES_MARKER}:
            return base64.b64decode(value[_BYTES_MARKER])
        return {key: decode_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    return value


def append_dim(dataset: xr.Dataset) -> str | None:
    """
    Find the shared leading dimension used for incremental snapshots.

    All data variables must have the same leading dimension. Otherwise the
    dataset cannot be split into rows.

    Args:
        dataset (xr.Dataset): Dataset to inspect.

    Returns:
        str | None: The shared leading dimension, if one exists.

    """
    dims = [var.dims for var in dataset.data_vars.values()]
    if not dims or any(not d for d in dims):
        return None
    first = dims[0][0]
    if any(d[0] != first for d in dims):
        return None
    return first


def row_frontier(dataset: xr.Dataset, dim: str) -> int:
    """
    Count the written rows along ``dim``.

    Preallocated floating-point grids use NaN for unwritten rows. The last row
    containing a finite value defines the frontier. Integer and boolean arrays
    cannot represent unwritten rows, so their full length is returned.

    Args:
        dataset (xr.Dataset): Dataset to inspect.
        dim (str): Dimension to measure along.

    Returns:
        int: Number of leading rows containing data.

    """
    total = int(dataset.sizes.get(dim, 0))
    if total == 0:
        return 0

    frontier = 0
    for var in dataset.data_vars.values():
        if dim not in var.dims:
            continue
        if var.dtype.kind not in "fc":
            return total
        values = var.transpose(dim, ...).values
        finite = np.isfinite(values)
        if finite.ndim > 1:
            finite = finite.any(axis=tuple(range(1, finite.ndim)))
        written = np.flatnonzero(finite)
        if written.size:
            frontier = max(frontier, int(written[-1]) + 1)
    return frontier


def snapshot_payload(dataset: xr.Dataset, *, binary: bool = False) -> dict[str, Any]:
    """
    Serialize an xarray dataset into the live snapshot representation.

    The server uses the binary form. The JSON-only form remains available for
    transports that cannot carry binary frames.

    Args:
        dataset (xr.Dataset): Dataset snapshot to serialize.
        binary (bool): If ``True``, numeric and boolean variables are left out
            of ``data`` and listed in ``binary_vars`` for
            :func:`pack_snapshot`. Complex, datetime, byte-string, and object
            arrays remain JSON-encoded.

    Returns:
        dict[str, Any]: The snapshot fields. With ``binary=True``, also
            includes ``binary_vars``.

    """
    coordinate_names = [str(name) for name in dataset.coords]
    data_variable_names = [str(name) for name in dataset.data_vars]
    variable_names = coordinate_names + data_variable_names

    binary_vars: dict[str, dict[str, Any]] = {}
    data: dict[str, Any] = {}
    for name in variable_names:
        array = dataset[name].values
        if binary and array.dtype.kind in _PLAIN_ARRAY_KINDS:
            binary_vars[name] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        else:
            data[name] = json_compatible(array)

    payload = {
        "coords": coordinate_names,
        "data_vars": data_variable_names,
        "var_dims": {
            name: [str(dim) for dim in dataset[name].dims] for name in variable_names
        },
        "var_attrs": {
            name: json_compatible(dict(dataset[name].attrs)) for name in variable_names
        },
        "var_dtypes": {name: str(dataset[name].dtype) for name in variable_names},
        "data": data,
        "attrs": json_compatible(dict(dataset.attrs)),
    }
    if binary:
        payload["binary_vars"] = binary_vars
    return payload


def pack_snapshot(payload: Mapping[str, Any], dataset: xr.Dataset) -> bytes:
    """
    Frame a ``binary=True`` snapshot payload as a single binary message.

    Args:
        payload (Mapping[str, Any]): Result of ``snapshot_payload(dataset,
            binary=True)``. Must carry a ``binary_vars`` field.
        dataset (xr.Dataset): Source of the arrays listed in ``binary_vars``.

    Returns:
        bytes: ``[4-byte header length][JSON header][concatenated raw arrays]``,
            ready to hand to a WebSocket ``send()`` as one binary frame.

    """
    header_bytes = json.dumps(payload, default=json_default).encode("utf-8")
    blob = b"".join(dataset[name].values.tobytes() for name in payload["binary_vars"])
    return _HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + blob


def unpack_snapshot(raw: bytes) -> dict[str, Any]:
    """
    Reverse :func:`pack_snapshot`.

    Args:
        raw (bytes): A message produced by :func:`pack_snapshot`.

    Returns:
        dict[str, Any]: Snapshot payload with binary variables restored as
            shaped, read-only NumPy arrays backed by ``raw``.

    Raises:
        ValueError: If a binary variable's declared shape and dtype do not
            match the bytes available for it.

    """
    (header_length,) = _HEADER_LENGTH.unpack_from(raw, 0)
    header_end = _HEADER_LENGTH.size + header_length
    payload = json.loads(raw[_HEADER_LENGTH.size : header_end])
    blob = raw[header_end:]

    offset = 0
    data = dict(payload.get("data", {}))
    for name, info in payload.get("binary_vars", {}).items():
        dtype = np.dtype(info["dtype"])
        shape = tuple(info["shape"])
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * dtype.itemsize
        if offset + nbytes > len(blob):
            raise ValueError(
                f"Binary snapshot truncated: '{name}' needs {nbytes} bytes at "
                f"offset {offset}, but only {len(blob) - offset} remain"
            )
        data[name] = np.frombuffer(
            blob, dtype=dtype, count=count, offset=offset
        ).reshape(shape)
        offset += nbytes

    payload["data"] = data
    return payload


def measurement_from_snapshot(snapshot: dict[str, Any]) -> xr.Dataset:
    """
    Reconstruct the measurement carried by a live snapshot response.

    Args:
        snapshot (dict[str, Any]): Snapshot returned by a live server.

    Returns:
        xr.Dataset: Dataset with variable metadata restored. Binary variables
            are read-only; copy them before modification.

    """
    coordinate_names: list[str] = snapshot.get("coords", [])
    data_variable_names: list[str] = snapshot.get("data_vars", [])
    variable_dimensions: dict[str, list[str]] = snapshot.get("var_dims", {})
    variable_attributes: dict[str, dict[str, Any]] = snapshot.get("var_attrs", {})
    variable_dtypes: dict[str, str] = snapshot.get("var_dtypes", {})
    values: dict[str, Any] = snapshot.get("data", {})

    def array_for(name: str) -> np.ndarray:
        """
        Decode one serialized variable into a NumPy array.

        Args:
            name (str): Variable name present in the snapshot data mapping.

        Returns:
            np.ndarray: Decoded variable values.

        """
        raw_value = values[name]
        # Binary variables are already decoded by unpack_snapshot.
        if isinstance(raw_value, np.ndarray):
            return raw_value

        decoded = decode_json_value(raw_value)
        dtype = variable_dtypes.get(name)
        try:
            return np.asarray(decoded, dtype=dtype)
        except (TypeError, ValueError):
            return np.asarray(decoded)

    coordinates: dict[str, Any] = {}
    for name in coordinate_names:
        if name not in values:
            continue
        array = array_for(name)
        dimensions = variable_dimensions.get(name)
        coordinates[name] = (
            (tuple(dimensions), array) if dimensions is not None else array
        )

    data_variables: dict[str, Any] = {}
    for name in data_variable_names:
        if name not in values:
            continue
        array = array_for(name)
        dimensions = variable_dimensions.get(name)
        if dimensions is None:
            dimensions = coordinate_names if array.ndim == len(coordinate_names) else []
        data_variables[name] = (tuple(dimensions), array)

    dataset = xr.Dataset(
        data_vars=data_variables,
        coords=coordinates,
        attrs=decode_json_value(snapshot.get("attrs", {})),
    )
    for name, attributes in variable_attributes.items():
        if name in dataset.variables:
            dataset[name].attrs.update(decode_json_value(attributes))
    return dataset
