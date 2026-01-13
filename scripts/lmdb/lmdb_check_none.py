import argparse
import lmdb
import pickle
import sys
from typing import Any, Dict, Optional

import numpy as np

# Optional known latent sizes for quick shape checks (float32 assumed)
LATENT_SIZES = {
    "S2L2A_B02_B03_B04_B08": 24,
    "S2L2A_B05_B06_B07_B11_B12_B8A": 12,
    "S2L2A_B01_B09": 4,
    "S2L1C_B02_B03_B04_B08": 24,
    "S2L1C_B05_B06_B07_B11_B12_B8A": 12,
    "S2L1C_B01_B09_B10": 4,
    "S2L1C_cloud_mask": 24,
    "S1RTC_vh_vv": 24,
    "DEM_DEM": 8,
    "LULC_LULC": 24,
}


def decode_value_safely(raw_value: memoryview) -> Optional[Any]:
    try:
        # memoryview is pickle-loadable; avoids extra copy
        return pickle.loads(raw_value)
    except Exception:
        return None


def validate_modalities(
    obj: Any,
    check_shapes: bool,
    max_shape_errors: int,
) -> Dict[str, int]:
    stats = {
        "none_keys": 0,
        "none_values": 0,
        "non_dict": 0,
        "empty_dict": 0,
        "unknown_modalities": 0,
        "shape_mismatch": 0,
    }

    if not isinstance(obj, dict):
        stats["non_dict"] += 1
        return stats

    if len(obj) == 0:
        stats["empty_dict"] += 1

    shape_errors_reported = 0

    for modality, data in obj.items():
        if modality is None:
            stats["none_keys"] += 1
        if data is None:
            stats["none_values"] += 1
            continue
        if not check_shapes:
            continue

        # Optional shape validation for known modalities
        latent = LATENT_SIZES.get(modality)
        if latent is None:
            stats["unknown_modalities"] += 1
            continue
        try:
            # frombuffer is zero-copy; cheap
            arr = np.frombuffer(data, dtype=np.float32)
            expected_size = 16 * latent * latent
            if arr.size != expected_size:
                stats["shape_mismatch"] += 1
                shape_errors_reported += 1
                if shape_errors_reported >= max_shape_errors:
                    # Avoid spending time formatting too many messages upstream
                    pass
        except Exception:
            stats["shape_mismatch"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an LMDB and ensure there are no None keys or None modality values. "
            "Optimized for large databases: minimal per-entry work, errors-only logging."
        )
    )
    parser.add_argument("lmdb_path", help="Path to the LMDB directory")
    parser.add_argument(
        "--stop-on-first",
        action="store_true",
        help="Stop scanning after the first detected issue (fast fail).",
    )
    parser.add_argument(
        "--check-shapes",
        action="store_true",
        help=(
            "Additionally check byte lengths against expected float32 shapes for known modalities."
        ),
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=50,
        help="Max number of issues to print (errors still counted).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=0,
        help="If > 0, print a progress message every N entries.",
    )
    parser.add_argument(
        "--readahead",
        action="store_true",
        help="Enable OS readahead (disabled by default to reduce cache thrashing).",
    )

    args = parser.parse_args()

    env = lmdb.open(
        args.lmdb_path,
        readonly=True,
        lock=False,
        readahead=args.readahead,
        max_readers=1,
        meminit=False,
    )

    total_entries = 0
    issues_found = 0
    issues_printed = 0

    # Aggregate stats for quick end-of-run summary
    agg_stats = {
        "none_lmdb_keys": 0,
        "empty_lmdb_keys": 0,
        "unpickle_failures": 0,
        "value_is_none": 0,
        "non_dict": 0,
        "empty_dict": 0,
        "none_modality_keys": 0,
        "none_modality_values": 0,
        "unknown_modalities": 0,
        "shape_mismatch": 0,
    }

    def maybe_print_issue(msg: str) -> None:
        nonlocal issues_printed
        if issues_printed < args.max_report:
            print(msg)
            issues_printed += 1

    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key, raw_value in cursor:
            total_entries += 1

            # LMDB returns bytes for key; None keys should not occur in practice,
            # but we still check to satisfy the requirement strictly.
            if key is None:
                issues_found += 1
                agg_stats["none_lmdb_keys"] += 1
                maybe_print_issue(f"Entry {total_entries}: LMDB key is None")
                if args.stop_on_first:
                    break
            elif len(key) == 0:
                issues_found += 1
                agg_stats["empty_lmdb_keys"] += 1
                maybe_print_issue(f"Entry {total_entries}: LMDB key is empty bytes")
                if args.stop_on_first:
                    break

            if raw_value is None:
                issues_found += 1
                agg_stats["value_is_none"] += 1
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: value is None")
                if args.stop_on_first:
                    break
                continue

            obj = decode_value_safely(raw_value)
            if obj is None:
                issues_found += 1
                agg_stats["unpickle_failures"] += 1
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: failed to unpickle value")
                if args.stop_on_first:
                    break
                continue

            if obj is None:
                issues_found += 1
                agg_stats["value_is_none"] += 1
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: unpickled object is None")
                if args.stop_on_first:
                    break
                continue

            stats = validate_modalities(obj, check_shapes=args.check_shapes, max_shape_errors=5)

            if stats["non_dict"]:
                issues_found += stats["non_dict"]
                agg_stats["non_dict"] += stats["non_dict"]
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: value is not a dict; cannot check modalities")
                if args.stop_on_first:
                    break

            if stats["empty_dict"]:
                issues_found += stats["empty_dict"]
                agg_stats["empty_dict"] += stats["empty_dict"]
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: dict has no modalities")
                if args.stop_on_first:
                    break

            if stats["none_keys"]:
                issues_found += stats["none_keys"]
                agg_stats["none_modality_keys"] += stats["none_keys"]
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: contains modality with None key")
                if args.stop_on_first:
                    break

            if stats["none_values"]:
                issues_found += stats["none_values"]
                agg_stats["none_modality_values"] += stats["none_values"]
                try:
                    key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                except Exception:
                    key_repr = str(key)
                maybe_print_issue(f"Key {key_repr}: contains modality with None value")
                if args.stop_on_first:
                    break

            if args.check_shapes:
                if stats["unknown_modalities"]:
                    agg_stats["unknown_modalities"] += stats["unknown_modalities"]
                if stats["shape_mismatch"]:
                    issues_found += stats["shape_mismatch"]
                    agg_stats["shape_mismatch"] += stats["shape_mismatch"]
                    try:
                        key_repr = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
                    except Exception:
                        key_repr = str(key)
                    maybe_print_issue(
                        f"Key {key_repr}: {stats['shape_mismatch']} modality shape/length mismatches"
                    )
                    if args.stop_on_first:
                        break

            if args.report_every and (total_entries % args.report_every == 0):
                print(f"Scanned {total_entries} entries so far; issues detected: {issues_found}")

    # Final report
    print("Scan complete.")
    print(f"Entries scanned: {total_entries}")
    print(f"Issues found: {issues_found}")
    if issues_found == 0:
        print("Status: OK — no None keys or values detected.")
    else:
        print("Status: FAILED — see counts below and logs above.")

    # Detailed counts
    for k in (
        "none_lmdb_keys",
        "empty_lmdb_keys",
        "unpickle_failures",
        "value_is_none",
        "non_dict",
        "empty_dict",
        "none_modality_keys",
        "none_modality_values",
        "unknown_modalities",
        "shape_mismatch",
    ):
        print(f"{k}: {agg_stats[k]}")

    return 0 if issues_found == 0 else 1


if __name__ == "__main__":
    sys.exit(main())