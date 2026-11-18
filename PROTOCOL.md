# Qimchi-Connect protocol

**Version: Qimchi-Connect v0.4.***

This document describes how a measurement program (the **producer**) publishes
a live measurement and how a viewer (the **client**), such as Qimchi, discovers
and reads it. It provides enough detail to implement a compatible producer or
client without using the `qimchi-connect` Python package.

The protocol has two parts:

1. A **discovery registry**, stored in an SQLite database on the shared
   machine, where producers list their published measurements.
2. A **WebSocket server** in each producer process, from which clients request
   measurement data.

A client reads the registry to find available measurements and their servers,
then requests current data when needed. Producers do not push data; clients
always request it.

Protocol name: `qimchi-connect`. Protocol version: `1`.

## Measurements and their data

A measurement is identified by a **measurement ID**, a non-empty string chosen
by the producer. It must be unique within the registry. Registering an existing
ID replaces its record.

A measurement is represented by an **xarray Dataset** containing named
variables with dimensions, data types, attributes, and values. Variables are
either coordinates or data variables, and the dataset may have its own
attributes. A snapshot captures the dataset state at request time.

## Discovery registry

### Location

The registry is the SQLite file `live_measurements.db` in the directory named
by the environment variable `QIMCHI_HOME`, or in `~/.qimchi` when it is unset.
Producers and clients must use the same directory. They can either run as the
same user or set `QIMCHI_HOME` to a shared location. Create the directory if it
does not exist.

### Table

```sql
CREATE TABLE IF NOT EXISTS live_measurements (
    measurement_id TEXT PRIMARY KEY,
    fpath TEXT NOT NULL,
    ws_url TEXT NOT NULL,
    ws_port INTEGER NOT NULL,
    live_status BOOLEAN NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    missed_since TEXT,
    last_seen TEXT
);
CREATE INDEX IF NOT EXISTS idx_live_status ON live_measurements(live_status);
```

Older registries lack `missed_since` and `last_seen`. Add them with
`ALTER TABLE ... ADD COLUMN` when they are missing.

| Column | Meaning |
|---|---|
| `measurement_id` | The unique measurement ID. |
| `fpath` | The saved measurement location, or `""` if it has not been saved. Clients use this location after the producer stops. It may be a path or a reference to part of a file, such as `run.nc#dt_path=/group` for an xarray DataTree group. |
| `ws_url` | The producer's WebSocket server, for example `ws://localhost:51234`. |
| `ws_port` | The port specified by `ws_url`. |
| `live_status` | `1` while the measurement is active; `0` after it ends. |
| `started_at` | When the measurement was registered. |
| `ended_at` | When the measurement ended, or `NULL`. |
| `last_seen` | The producer's most recent heartbeat, or `NULL` if it sends none. |
| `missed_since` | When a client first failed to reach a producer that does not send heartbeats. Cleared when the producer responds again. |

All timestamps are ISO 8601 text in UTC with an explicit offset, as written by
Python's `datetime.now(timezone.utc).isoformat()`, for example
`2026-11-18T17:05:09.123456+00:00`.

Open the database with a busy timeout (10 seconds in the Python package), and
keep transactions short because multiple producers and clients may access it
concurrently.

### Producer updates

1. **Register**: start the WebSocket server first, then
   `INSERT OR REPLACE` a row with `live_status = 1`, `ended_at = NULL` and the
   current time as `started_at`.
2. **Heartbeat** (optional but recommended): set `last_seen` to the current
   time every 5 seconds. Send the first heartbeat immediately after
   registering.
3. **Saved location**: update `fpath` whenever the saved location changes. A
   producer that writes intermediate files before the final file updates
   `fpath` for each location.
4. **End**: set `live_status = 0` and `ended_at` to the current time, then
   stop answering for that measurement.

### Determining liveness

A producer may stop without ending its rows, for example if its process
crashes. Clients therefore mark stale rows as ended. Producers perform the
same cleanup before registering. For each row with `live_status = 1`:

- If `last_seen` is within the last 15 seconds (three heartbeats), the
  producer is alive.
- If `last_seen` is older, mark the row as ended.
- If `last_seen` is `NULL`, send `list` to its `ws_url`. If the reply does not
  include the measurement, end the row. If the server cannot be reached, set
  `missed_since` (if unset) and end the row once it has been unreachable for
  60 seconds.

Ended rows older than 7 days may be deleted.

A client listing live measurements returns rows with `live_status = 1`, except
those whose `last_seen` is older than 15 seconds.

## WebSocket server

Each producer process runs one WebSocket server, which may serve several
measurements. A client may send several requests over one connection; the
Python client opens a new connection for each request.

- Each **request** is one JSON object sent as a text message.
- Each request receives exactly one **response**, in request order.
- A successful `get_snapshot` response is one **binary** message (see
  [Snapshot messages](#snapshot-messages)). Every other response, including a
  failed `get_snapshot`, is one JSON object sent as a text message.
- Messages may be up to 1000 MiB (1 048 576 000 bytes). Configure both ends
  to accept that size.
- The server does not use permessage-deflate compression because numeric data
  typically compresses poorly.

### Requests

Every request has an `action`. Requests about one measurement also carry
`measurement_id`. Clients may add `protocol_version`; servers ignore it.

| `action` | Other fields | Purpose |
|---|---|---|
| `capabilities` | | Describe the server. |
| `list` | | List the measurements this server publishes. |
| `get_snapshot` | `measurement_id`, optional `since_rows` | The measurement's current data. |
| `get_data` | `measurement_id`, optional `variables` | The measurement structure, or selected variable values, as JSON. |
| `get_zarr_array` | `measurement_id`, `array_path` | One variable with its dimensions, dtype and attributes, as JSON. |

### Responses

Every response has `success` (`true` or `false`). Failed responses include an
`error` message. Responses for a specific measurement include its
`measurement_id`. Responses to valid requests also include
`"protocol": "qimchi-connect"` and `"protocol_version": 1`. A message that is
not valid JSON receives
`{"success": false, "error": "Invalid JSON"}`.

**`capabilities`**

```json
{
  "success": true,
  "protocol": "qimchi-connect",
  "protocol_version": 1,
  "actions": ["capabilities", "list", "get_data", "get_snapshot", "get_zarr_array"],
  "id_field": "measurement_id",
  "snapshot_format": "xarray-binary"
}
```

**`list`**: `measurements` is a list of measurement IDs.

**`get_data`**: without `variables` (or with an empty list), the response
contains `coords` and `data_vars` (lists of names), `attrs`, and `source`. With
`variables`, it contains `data`, which maps each existing requested variable
to its values as nested JSON lists, and `attrs`.

**`get_zarr_array`**: `array_path` is a variable name. The response contains
`array_path`, `data` (nested JSON lists), `shape`, `dtype`, `dims`, and `attrs`.

**`get_snapshot`**: see the next section.

A request for a measurement the server does not publish fails with
`Measurement <id> not found`.

### Snapshot messages

A successful `get_snapshot` response is one binary message:

```
+-------------------+---------------------------+---------------------------------+
| header length (4) | header (UTF-8 JSON)       | arrays, in binary_vars order    |
+-------------------+---------------------------+---------------------------------+
```

- **Header length**: an unsigned 32-bit big-endian integer containing the
  header size in bytes.
- **Header**: a JSON object with the response fields and the dataset's
  structure.
- **Arrays**: the values of each variable named in `binary_vars`, in key order.
  Each array uses row-major (C) order, with no padding between arrays.

Header fields:

| Field | Meaning |
|---|---|
| `success`, `measurement_id`, `protocol`, `protocol_version` | As for every response. |
| `source` | Producer metadata as a free-form JSON object. Conventional keys are `source_package` (such as `qanary`, `qcodes`, `quantify`, or `elicit`) and `source_format` (such as `zarr`, `netcdf`, or `hdf5`). |
| `coords` | Names of the coordinate variables. |
| `data_vars` | Names of the data variables. |
| `var_dims` | For every variable, the list of its dimension names. |
| `var_dtypes` | For every variable, its NumPy dtype name, such as `float64`. |
| `var_attrs` | For every variable, its attributes. |
| `attrs` | The dataset's attributes. |
| `binary_vars` | For every variable stored in the binary part, `{"dtype": ..., "shape": [...]}`. |
| `data` | For every other variable, its values as nested JSON lists. |
| `append_dim`, `rows_from`, `rows_written`, `rows_total` | Row fields; see [Fetching only new rows](#fetching-only-new-rows). |

Boolean, signed and unsigned integer, and floating-point variables (NumPy
kinds `b`, `i`, `u`, and `f`) are stored in the binary section using the
producer's native byte order. Every platform supported by Qimchi is
little-endian. Other variables, including complex numbers, strings,
timestamps, and Python objects, are stored in `data`.

**JSON values.** Values in `data` and in attributes follow these rules:

- A complex number is `{"__qimchi_connect_complex__": [real, imaginary]}`.
- Bytes are `{"__qimchi_connect_bytes__": "<base64>"}`.
- Dates and times are ISO 8601 strings.
- Tuples and sets become lists.
- Floating-point attributes may contain the non-standard JSON tokens `NaN`,
  `Infinity` and `-Infinity`. Parse the header with a decoder that accepts
  them.

**Rebuilding the dataset.** Create each coordinate and then each data variable
from its values, `var_dims` and `var_dtypes`, then apply `var_attrs` and
`attrs`, restoring complex numbers and bytes from their markers.

By default, the server creates at most one snapshot per measurement every 0.25
seconds. Requests within that interval receive the cached snapshot.

### Fetching only new rows

Many measurements append data one row at a time. A client may request only the
rows added since its previous request.

A measurement supports incremental snapshots when every data variable shares
the same first dimension, called the **append dimension**, and previously
written rows never change. A full snapshot of such a measurement includes:

| Field | Meaning |
|---|---|
| `append_dim` | The append dimension. |
| `rows_from` | Index of the first row in this response: `0` for a full snapshot. |
| `rows_written` | How many leading rows contain data. |
| `rows_total` | How many rows the dataset has, including rows not yet written. |

`rows_written` is one past the last row where any floating-point data variable
has a finite value. Later rows are unwritten space, filled with NaN and
reserved for the rest of the measurement. If a data variable cannot represent
NaN, as with integer and Boolean variables, every row counts as written.

To fetch only new rows, send `since_rows` with the number of rows the client
already holds, normally the previous `rows_written`. If
`0 < since_rows <= rows_written` (and `since_rows <= rows_total`), the response
contains only rows `since_rows` to `rows_written - 1` along the append
dimension, with `rows_from = since_rows`. Variables without the append
dimension are sent whole. The client merges these rows into the dataset it
holds; the server keeps no state about the client.

Otherwise, the response is a full snapshot with `rows_from = 0`, and the
client should replace its current dataset. The client should also restart from
a full snapshot when `rows_total` or the variables differ from its current
dataset because the measurement structure has changed.

If a measurement changes previously written rows—for example, by averaging
repeated sweeps into the same rows—the producer must mark it as **not
append-only** (`append_only=False` in the Python package). Its snapshots omit
the row fields, and the server ignores `since_rows` and sends the complete
dataset.

A server may ignore `since_rows`; versions before Qimchi-Connect 0.3 did so. A
response without `rows_from` is a full snapshot and must not be treated as an
incremental update.

## Compatibility

New features use optional fields. Clients and servers must ignore unknown
fields and must not require fields marked optional in this document. Any
incompatible change requires a new `protocol_version`.
