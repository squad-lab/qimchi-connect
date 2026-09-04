"""
Command-line maintenance utilities for the live discovery registry.

"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from qimchi_connect.registry import get_database_path, maintain_registry


def _version() -> str:
    """
    Return the installed package version.

    Returns:
        str: Distribution version, or ``"unknown"`` when the package metadata
            is not installed, as in a source checkout run without an install.

    """
    try:
        return version("qimchi-connect")
    except PackageNotFoundError:
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.

    """
    parser = argparse.ArgumentParser(
        prog="qimchi-connect",
        description=(
            "Maintain the live measurement discovery registry that producers "
            "register with and Qimchi reads."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"qimchi-connect {_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cleanup = subparsers.add_parser(
        "cleanup",
        help="repair and prune the registry",
        description=(
            "Probe every endpoint still recorded as live, mark the records it "
            "no longer advertises as ended, then delete ended records past "
            "the retention period."
        ),
    )
    cleanup.add_argument(
        "--retention-days",
        type=int,
        default=7,
        help="delete ended records older than this many days (default: %(default)s)",
    )
    cleanup.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help=(
            "seconds allowed for each endpoint probe, applied to the "
            "connection and to the response separately (default: %(default)s)"
        ),
    )
    cleanup.add_argument(
        "--retries",
        type=int,
        default=2,
        help="probe attempts before an endpoint counts as unreachable "
        "(default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run a live-registry maintenance command.

    Args:
        argv (Sequence[str] | None): Arguments excluding the executable name.

    Returns:
        int: Process exit status.

    """
    arguments = _parser().parse_args(argv)
    if arguments.command == "cleanup":
        result = maintain_registry(
            retention_days=arguments.retention_days,
            timeout=arguments.timeout,
            retries=arguments.retries,
        )
        print(f"Registry: {get_database_path()}")
        print(f"Marked stale: {len(result.stale_measurement_ids)}")
        print(f"Deleted expired: {result.deleted_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
