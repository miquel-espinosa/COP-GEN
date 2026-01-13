#!/usr/bin/env python3
"""lmdb_inspect.py

Quickly inspect the contents of an LMDB database.

Usage
-----
$ python scripts/lmdb_inspect.py /path/to/lmdb_folder --max 5

The script prints a summary of the first *N* entries (default: 10), including
key names, value types, and for NumPy arrays their shape & dtype.
"""
from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path
from typing import Any

import lmdb  # type: ignore
import numpy as np
import zarr  # type: ignore
import zipfile
import tempfile
import os

from encode_moments_vae import load_zarr_zip_blob


def _decode_key(raw_key: bytes) -> str:
    """Decode a bytes key to str, falling back to repr if needed."""
    try:
        return raw_key.decode()
    except Exception:
        return repr(raw_key)


def _try_load_array(raw: bytes) -> tuple[bool, Any]:
    """Attempt to load a NumPy array from *raw* bytes.

    Returns (success, obj).  If *success* is True, *obj* is the ndarray.
    Otherwise *obj* is the exception raised by np.load.
    """
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=True)
        return True, arr
    except Exception as exc:  # pylint: disable=broad-except
        return False, exc


def _print_array(arr: np.ndarray, indent: str = "    ") -> None:
    print(f"{indent} array – shape: {arr.shape}, dtype: {arr.dtype}, type: {type(arr)}")


def _inspect_value(value: bytes, indent: str = "    ", uses_zarr: bool = False) -> None:
    """Pretty-print the structure of one LMDB value."""
    # If the data was stored via Zarr, try that first – it's the fastest path
    if uses_zarr:
        maybe_arr = load_zarr_zip_blob(value)
        # Convert zarr to numpy array
        # maybe_arr = maybe_arr[:]
        _print_array(maybe_arr, indent)
        return

    # Otherwise attempt to unpickle the raw bytes (default for many workflows)
    try:
        obj = pickle.loads(value)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"{indent}❌ Could not unpickle value: {exc}")
        success, maybe_arr = _try_load_array(value)
        if success:
            _print_array(maybe_arr, indent)
        else:
            print(f"{indent}Raw bytes length: {len(value)}")
        return

    # Handle different top-level object types
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{indent}Sub-key: {k}")
            if isinstance(v, (bytes, bytearray)):
                success, arr_or_exc = _try_load_array(v)
                if success:
                    _print_array(arr_or_exc, indent + "    ")
                else:
                    print(f"{indent}    ⚠️ Could not decode sub-value: {arr_or_exc}")
            else:
                print(f"{indent}    Type: {type(v)} – preview: {str(v)[:60]}")
    elif isinstance(obj, np.ndarray):
        _print_array(obj, indent)
    else:
        print(f"{indent}Type: {type(obj)} – preview: {str(obj)[:80]}")


def inspect_lmdb(path: Path, max_entries: int = 10, *, exact_key: str | None = None, contains: str | None = None, uses_zarr: bool = False) -> None:
    """Inspect an LMDB database.

    Parameters
    ----------
    path: Path
        Folder containing ``data.mdb`` and ``lock.mdb``.
    max_entries: int, default 10
        Maximum number of entries to show when *exact_key* is not provided.
    exact_key: str | None
        If given, retrieve *only* this key (exact match) and display its value.
    contains: str | None
        If given, iterate through the DB and show up to *max_entries* keys that
        contain this substring (case-sensitive).
    """
    if not path.exists():
        sys.exit(f"❌ Path does not exist: {path}")

    # Open the environment in read-only mode (no locks required)
    env = lmdb.open(str(path), readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()

        # ------------------------------------------------------------------
        # Exact key lookup
        # ------------------------------------------------------------------
        if exact_key is not None:
            raw_value = txn.get(exact_key.encode())
            if raw_value is None:
                print(f"⚠️ Key not found: {exact_key}")
                return

            print(f"Key: {exact_key}")
            try:
                _inspect_value(raw_value, uses_zarr=uses_zarr)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"    ⚠️ Error while inspecting value: {exc}")
            return  # Nothing else to do

        # ------------------------------------------------------------------
        # Iteration mode (with optional substring filter)
        # ------------------------------------------------------------------
        match_idx = 0
        for raw_key, raw_value in cursor:
            key_str = _decode_key(raw_key)

            if contains is not None and contains not in key_str:
                continue  # Skip non-matching keys

            if match_idx >= max_entries:
                break

            print(f"Entry {match_idx} – Key: {key_str}")
            try:
                _inspect_value(raw_value, uses_zarr=uses_zarr)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"    ⚠️ Error while inspecting value: {exc}")
            print()
            match_idx += 1

        if match_idx == 0:
            note = f"containing '{contains}'" if contains else "in DB"
            print(f"⚠️ No keys {note} found.")


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect an LMDB database. Either show the first N entries, "
        "look up a specific key, or list keys containing a given substring.",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to LMDB folder (directory containing data.mdb & lock.mdb)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--key",
        "-k",
        type=str,
        help="Exact key to retrieve and display",
    )
    group.add_argument(
        "--contains",
        "-c",
        type=str,
        help="Substring to filter keys (case-sensitive)",
    )
    
    parser.add_argument(
        "--uses_zarr",
        "-z",
        action="store_true",
        default=False,
        help="Indicates that array payloads were serialized via Zarr (ZipStore)",
    )

    parser.add_argument(
        "--max",
        "-n",
        type=int,
        default=10,
        help="Maximum number of entries to display when iterating (default: 10)",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()
    inspect_lmdb(
        Path(args.path),
        max_entries=args.max,
        exact_key=args.key,
        contains=args.contains,
        uses_zarr=args.uses_zarr,
    )


if __name__ == "__main__":
    main()