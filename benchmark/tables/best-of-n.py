from __future__ import annotations

import argparse
import re
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Constants and normalizers
# ---------------------------

COMMON_MODALITIES_ORDER: List[str] = [
    "lat_lon",  # will also map 'coords' -> 'lat_lon'
    "DEM",
    "LULC",
    "S1RTC",
    "S2L1C",
    "S2L2A",
]
COMMON_MODALITIES: set[str] = set(COMMON_MODALITIES_ORDER)
IGNORED_OUTPUTS: set[str] = {"timestamps"}  # always ignore timestamps in comparisons

METRICS_FOR_TARGET_TABLE: Dict[str, List[str]] = {
    "DEM": ["mae", "rmse", "ssim", "psnr"],
    "S1RTC": ["mae", "rmse", "ssim", "psnr"],
    "S2L1C": ["mae", "rmse", "ssim", "psnr"],
    "S2L2A": ["mae", "rmse", "ssim", "psnr"],
    "LULC": ["overall_top1", "overall_top3", "mean_iou", "fw_iou", "mean_f1"],
    "lat_lon": ["median_km", "mean_km", "std_km", "rmse_km"],
}

HIGHER_BETTER_METRICS: set[str] = {
    "ssim",
    "psnr",
    "overall_top1",
    "overall_top3",
    "mean_iou",
    "fw_iou",
    "mean_f1",
}
LOWER_BETTER_METRICS: set[str] = {
    "mae",
    "rmse",
    "median_km",
    "mean_km",
    "std_km",
    "rmse_km",
}


def normalize_modality_name(name: str) -> str:
    n = name.strip()
    if n == "coords":
        return "lat_lon"
    return n


# ---------------------------
# Experiment name parsing
# ---------------------------

@dataclass
class ParsedExperiment:
    model: str  # "copgen" or "terramind"
    run_dir: Path
    inputs: List[str]
    outputs: List[str]
    seed: Optional[int]


def _merge_multiword_tokens(parts: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        # Merge 'cloud' + 'mask'
        if p == "cloud" and i + 1 < len(parts) and parts[i + 1] == "mask":
            merged.append("cloud_mask")
            i += 2
            continue
        # Merge 'lat' + 'lon'
        if p == "lat" and i + 1 < len(parts) and parts[i + 1] == "lon":
            merged.append("lat_lon")
            i += 2
            continue
        merged.append(p)
        i += 1
    return merged


def _split_modalities_str(s: str) -> List[str]:
    # Split by underscore then repair known multiword tokens
    raw = [p for p in s.split("_") if p]
    merged = _merge_multiword_tokens(raw)
    # Normalize coords/lat_lon mapping and uppercase known codes
    normalized = [normalize_modality_name(p) for p in merged]
    return normalized


def parse_experiment_name(dir_name: str, model: str) -> Tuple[List[str], List[str], Optional[int]]:
    """
    Supports:
      - input_<A_B_C>_output_<X_Y>_seed_<S>
      - <X>_from_<A_B_C>_seed_<S>  (seen in some scripts; we still support for robustness)
    Returns (inputs, outputs, seed)
    """
    seed: Optional[int] = None
    seed_match = re.search(r"_seed_(\d+)", dir_name)
    if seed_match:
        seed = int(seed_match.group(1))

    if "input_" in dir_name and "_output_" in dir_name:
        i0 = dir_name.find("input_") + len("input_")
        i1 = dir_name.find("_output_")
        inputs_str = dir_name[i0:i1]
        # outputs may run until _seed_ or end
        o0 = i1 + len("_output_")
        o1 = dir_name.find("_seed_", o0)
        outputs_str = dir_name[o0:] if o1 == -1 else dir_name[o0:o1]
        inputs = _split_modalities_str(inputs_str)
        outputs = _split_modalities_str(outputs_str)
        return inputs, outputs, seed

    # Fallback: "<OUT>_from_<IN1_IN2>_seed_<S>"
    if "_from_" in dir_name:
        o1 = dir_name.find("_from_")
        out_str = dir_name[:o1]
        i0 = o1 + len("_from_")
        i1 = dir_name.find("_seed_", i0)
        inputs_str = dir_name[i0:] if i1 == -1 else dir_name[i0:i1]
        inputs = _split_modalities_str(inputs_str)
        outputs = _split_modalities_str(out_str)
        return inputs, outputs, seed

    # Unknown pattern - assume nothing
    return [], [], seed


# ---------------------------
# Metrics parsing
# ---------------------------

_kv_key_re = re.compile(r"([A-Za-z0-9_:@]+)=")
_chunk_re = re.compile(r"\[([^\]]+)\]\s+([\s\S]*?)(?=(?:\n\[)|\Z)")


def _normalize_metric_key(raw_key: str) -> Optional[str]:
    # band:B01:mae -> B01_mae, band:vh:psnr -> vh_psnr
    if raw_key.startswith("band:"):
        parts = raw_key.split(":")
        if len(parts) == 3:
            _, band, metric = parts
            # Skip band:DEM:* duplicates
            if band == "DEM":
                return None
            return f"{band}_{metric}"
    # acc@1km -> acc_at_1km
    if "@" in raw_key:
        return raw_key.replace("@", "_at_")
    return raw_key


def _parse_value(v: str):
    v = v.strip()
    # strip trailing separators that may leak in parsing
    v = v.rstrip("|,")
    if len(v) == 0:
        return np.nan
    if v.startswith("[") and v.endswith("]"):
        try:
            arr = ast.literal_eval(v)
            # ensure list of floats
            return [float(x) for x in arr]
        except Exception:
            return np.nan
    try:
        return float(v)
    except ValueError:
        return v


def parse_output_metrics_file(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns: { modality -> { metric_key: value } }
    Where metric_key is normalized (e.g., B02_mae, overall_top1, acc_at_10km, ...).
    Arrays are expanded as metric_key_{i}.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    by_modality: Dict[str, Dict[str, float]] = {}
    for m in _chunk_re.finditer(text):
        modality_raw = m.group(1).strip()
        modality = normalize_modality_name(modality_raw)
        body = m.group(2)
        metrics: Dict[str, float] = {}
        # scan key=value tokens robustly
        key_spans = list(_kv_key_re.finditer(body))
        for idx, km in enumerate(key_spans):
            key_raw = km.group(1)
            v_start = km.end()
            v_end = key_spans[idx + 1].start() if idx + 1 < len(key_spans) else len(body)
            val_str = body[v_start:v_end].strip()
            norm_key = _normalize_metric_key(key_raw)
            if not norm_key:
                continue
            val = _parse_value(val_str)
            if isinstance(val, list):
                for i, x in enumerate(val):
                    metrics[f"{norm_key}_{i}"] = x
            else:
                # try float conversion for simple numbers
                try:
                    metrics[norm_key] = float(val)  # type: ignore[arg-type]
                except Exception:
                    # keep non-numeric if any (rare)
                    pass
        by_modality[modality] = metrics
    return by_modality


def parse_output_metrics_per_tile(path: Path) -> pd.DataFrame:
    """
    Parse per-tile metrics CSV.
    Returns a DataFrame with columns:
      tile_id, modality, <metric columns...> (band-level metrics are dropped)
    Column names are normalized to match the same convention as parse_output_metrics_file.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    # Keep only tile-level metrics (drop band:* columns)
    keep_cols = [c for c in df.columns if c in {"tile_id", "modality"} or not c.startswith("band:")]
    df = df[keep_cols]

    # Normalize modality names
    if "modality" in df.columns:
        df["modality"] = df["modality"].map(normalize_modality_name)

    rename_map: Dict[str, str] = {}
    metric_cols: List[str] = []
    for col in df.columns:
        if col in {"tile_id", "modality"}:
            continue
        norm = _normalize_metric_key(col)
        if norm and norm != col:
            rename_map[col] = norm
            metric_cols.append(norm)
        else:
            metric_cols.append(col)

    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce metric columns to numeric
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------
# Row building
# ---------------------------

def inputs_to_flags(inputs: Iterable[str]) -> Dict[str, int]:
    s = set(normalize_modality_name(x) for x in inputs)
    return {f"input_{m}": int(m in s) for m in COMMON_MODALITIES_ORDER}


def _condition_key_from_missing(missing: Iterable[str]) -> str:
    """
    Encode which (if any) conditioning modalities besides the target are absent.
    """
    missing_sorted = sorted(normalize_modality_name(m) for m in missing)
    filtered = [m for m in missing_sorted if m in COMMON_MODALITIES]
    if not filtered:
        return "all_except_target"
    return "missing_" + "+".join(filtered)


def _missing_from_condition_key(key: str) -> List[str]:
    if not key or key == "all_except_target":
        return []
    prefix = "missing_"
    if key.startswith(prefix):
        rest = key[len(prefix) :]
        if not rest:
            return []
        return [normalize_modality_name(m) for m in rest.split("+") if m]
    return []


def _variant_label(condition_key: str) -> str:
    missing = _missing_from_condition_key(condition_key)
    if not missing:
        return "All-minus-target"
    label = "No " + " + ".join(missing)
    return label


def _condition_sort_key(condition_key: str) -> Tuple[int, str]:
    if condition_key == "all_except_target":
        return (0, "")
    return (1, condition_key)


def build_rows_for_run(model: str, run_dir: Path) -> List[Dict]:
    """
    There may be multiple target rows per run (e.g., COP-GEN generating several outputs).
    """
    dir_name = run_dir.name
    inputs, outputs, seed = parse_experiment_name(dir_name, model)
    if not inputs and not outputs:
        return []
    # normalize modalities
    inputs_norm = [normalize_modality_name(x) for x in inputs]
    outputs_norm = [normalize_modality_name(x) for x in outputs]
    # locate metrics file
    metrics_path = run_dir / "output_metrics.txt"
    if not metrics_path.exists():
        alt = run_dir / "generations" / "output_metrics.txt"
        if alt.exists():
            metrics_path = alt
        else:
            # as last resort, search shallow
            found = list(run_dir.rglob("output_metrics.txt"))
            if found:
                metrics_path = found[0]
    if not metrics_path.exists():
        return []

    by_mod = parse_output_metrics_file(metrics_path)

    rows: List[Dict] = []
    input_set = set(inputs_norm)
    # Determine candidate targets: outputs that are in COMMON_MODALITIES and not ignored
    candidate_targets = [o for o in outputs_norm if o in COMMON_MODALITIES and o not in IGNORED_OUTPUTS]
    for target in candidate_targets:
        if target in input_set:
            # Skip degenerate runs where target is also supplied as input
            continue
        missing_extra = [m for m in COMMON_MODALITIES_ORDER if m not in input_set and m != target]
        condition_key = _condition_key_from_missing(missing_extra)

        metrics = by_mod.get(target, {})
        if not metrics:
            # No metrics for this target in this run
            continue

        row: Dict = {
            "model": model,
            "target": target,
            "condition_key": condition_key,
            "seed": seed if seed is not None else -1,
            **inputs_to_flags(inputs_norm),
        }
        # attach metrics
        for k, v in metrics.items():
            # numeric only
            try:
                row[k] = float(v)
            except Exception:
                pass
        rows.append(row)
    return rows


def _find_per_tile_metrics_path(run_dir: Path) -> Optional[Path]:
    """
    Locate output_metrics_per_tile.csv within a run directory.
    """
    candidates = [
        run_dir / "output_metrics_per_tile.csv",
        run_dir / "generations" / "output_metrics_per_tile.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # last resort: shallow search
    found = list(run_dir.rglob("output_metrics_per_tile.csv"))
    if found:
        return found[0]
    return None


def build_tile_rows_for_run(model: str, run_dir: Path) -> List[Dict]:
    """
    Build per-tile rows for a run, keeping configurations where the target
    modality is generated from the available conditioning inputs.
    Each row corresponds to a single tile_id for a target modality.
    """
    dir_name = run_dir.name
    inputs, outputs, seed = parse_experiment_name(dir_name, model)
    if not inputs and not outputs:
        return []
    inputs_norm = [normalize_modality_name(x) for x in inputs]
    outputs_norm = [normalize_modality_name(x) for x in outputs]

    metrics_path = _find_per_tile_metrics_path(run_dir)
    if metrics_path is None:
        return []

    tile_df = parse_output_metrics_per_tile(metrics_path)
    if tile_df.empty:
        return []

    rows: List[Dict] = []
    input_set = set(inputs_norm)
    candidate_targets = [o for o in outputs_norm if o in COMMON_MODALITIES and o not in IGNORED_OUTPUTS]
    for target in candidate_targets:
        if target in input_set:
            continue
        target_df = tile_df[tile_df["modality"] == target]
        if target_df.empty:
            continue
        missing_extra = [m for m in COMMON_MODALITIES_ORDER if m not in input_set and m != target]
        condition_key = _condition_key_from_missing(missing_extra)

        metric_cols = [c for c in target_df.columns if c not in {"tile_id", "modality"}]
        for _, r in target_df.iterrows():
            row: Dict = {
                "model": model,
                "target": target,
                "condition_key": condition_key,
                "seed": seed if seed is not None else -1,
                "tile_id": r["tile_id"],
                **inputs_to_flags(inputs_norm),
            }
            for col in metric_cols:
                val = r[col]
                try:
                    row[col] = float(val)
                except Exception:
                    pass
            rows.append(row)
    return rows


def scan_all_runs(dataset_root: Path) -> pd.DataFrame:
    """
    Walk <dataset_root>/outputs/{copgen,terramind}/* and build raw rows.
    """
    rows: List[Dict] = []
    base = dataset_root / "outputs"
    for model in ("copgen", "terramind"):
        model_dir = base / model
        if not model_dir.exists():
            continue
        for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            rows.extend(build_rows_for_run(model, run_dir))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def scan_all_runs_tiles(dataset_root: Path) -> pd.DataFrame:
    """
    Walk <dataset_root>/outputs/{copgen,terramind}/* and build per-tile rows.
    """
    rows: List[Dict] = []
    base = dataset_root / "outputs"
    for model in ("copgen", "terramind"):
        model_dir = base / model
        if not model_dir.exists():
            continue
        for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            rows.extend(build_tile_rows_for_run(model, run_dir))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


# ---------------------------
# Aggregation
# ---------------------------

GROUP_KEYS = [
    "model",
    "input_DEM",
    "input_LULC",
    "input_S1RTC",
    "input_S2L1C",
    "input_S2L2A",
    "input_lat_lon",
    "target",
    "condition_key",
]


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Identify metric columns: numeric but not input_* or seed
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if not c.startswith("input_") and c not in {"seed"}]
    # Group
    g = df.groupby(GROUP_KEYS, dropna=False)
    means = g[metric_cols].mean()
    stds = g[metric_cols].std(ddof=0)
    counts = g.size().rename("n_samples")
    # Flatten column names
    means.columns = [f"{c}_mean" for c in means.columns]
    stds.columns = [f"{c}_std" for c in stds.columns]
    out = pd.concat([means, stds, counts], axis=1).reset_index()
    return out


def aggregate_best_per_tile(tile_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each configuration (GROUP_KEYS), select the best metric per tile across seeds/runs,
    then aggregate those best-per-tile metrics (mean across tiles).
    """
    if tile_df.empty:
        return tile_df

    numeric_cols = tile_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if not c.startswith("input_") and c not in {"seed"}]

    # Best per tile (choose min or max depending on metric)
    group_keys_tile = GROUP_KEYS + ["tile_id"]
    agg_best = {}
    for m in metric_cols:
        if m in LOWER_BETTER_METRICS:
            agg_best[m] = "min"
        else:
            agg_best[m] = "max"
    best_per_tile = tile_df.groupby(group_keys_tile, dropna=False).agg(agg_best).reset_index()

    # Aggregate best-per-tile metrics across tiles (mean) and count tiles
    agg_mean = {m: "mean" for m in metric_cols}
    best_agg = best_per_tile.groupby(GROUP_KEYS, dropna=False).agg(agg_mean)
    counts = best_per_tile.groupby(GROUP_KEYS, dropna=False).size().rename("n_tiles")
    # Also report how many experiment runs were available for this configuration.
    # We treat runs as distinct seeds (tile_df contains "seed" but not run_dir).
    n_runs = tile_df.groupby(GROUP_KEYS, dropna=False)["seed"].nunique().rename("n_runs")
    out = pd.concat([best_agg, counts, n_runs], axis=1).reset_index()
    return out


def aggregate_tile_mean_std(tile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-tile metrics after first averaging across seeds for each tile,
    reporting mean and std across tiles for every configuration.
    """
    if tile_df.empty:
        return tile_df

    numeric_cols = tile_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if not c.startswith("input_") and c not in {"seed"}]
    if "tile_id" in metric_cols:
        metric_cols.remove("tile_id")
    if not metric_cols:
        return pd.DataFrame()

    # First average metrics across seeds for each tile/configuration pair
    group_keys_tile = GROUP_KEYS + ["tile_id"]
    tile_means = tile_df.groupby(group_keys_tile, dropna=False)[metric_cols].mean().reset_index()

    # Then aggregate across tiles to get dataset-level mean/std
    g = tile_means.groupby(GROUP_KEYS, dropna=False)
    means = g[metric_cols].mean()
    stds = g[metric_cols].std(ddof=0)
    n_tiles = g.size().rename("n_tiles")
    # Count distinct seeds that contributed before averaging
    n_runs = tile_df.groupby(GROUP_KEYS, dropna=False)["seed"].nunique().rename("n_runs")
    means.columns = [f"{c}_mean" for c in means.columns]
    stds.columns = [f"{c}_std" for c in stds.columns]
    out = pd.concat([means, stds, n_tiles, n_runs], axis=1).reset_index()
    return out


# ---------------------------
# CLI printing
# ---------------------------

def _format_mean_std(mean_val: Optional[float], std_val: Optional[float], float_fmt: str = ".4f") -> str:
    if mean_val is None or (isinstance(mean_val, float) and (np.isnan(mean_val))):
        return "-"
    if std_val is None or (isinstance(std_val, float) and (np.isnan(std_val))):
        return f"{mean_val:{float_fmt}}"
    return f"{mean_val:{float_fmt}} ± {std_val:{float_fmt}}"


def _find_row(
    agg: pd.DataFrame, model: str, target: str, condition_key: str, count_col: str = "n_samples"
) -> Optional[pd.Series]:
    if agg.empty:
        return None
    sel = agg[
        (agg["model"] == model)
        & (agg["target"] == target)
        & (agg["condition_key"] == condition_key)
    ]
    if sel.empty:
        return None
    # If multiple rows (unlikely), pick the one with the highest count column
    if count_col in sel.columns:
        sel = sel.sort_values(count_col, ascending=False)
    return sel.iloc[0]


def _condition_string(target: str, missing_extra: List[str]) -> str:
    excluded = set(missing_extra)
    excluded.add(target)
    included = [m for m in COMMON_MODALITIES_ORDER if m not in excluded]
    return " + ".join(included)


def print_summary_tables(agg: pd.DataFrame, float_fmt: str) -> None:
    # Print explanation of how results are formatted in the tables below
    print("\n" + "=" * 80)
    print("\033[31m=== EXPERIMENT LEVEL RESULTS ===\033[0m")
    print("BEST-OF-N CROSS-MODAL RESULTS")
    print("=" * 80)
    print("\nHow to read the tables:")
    print("  - Each row shows generating a target modality from the available inputs")
    print("  - 'Variant' indicates if extra conditioning modalities were omitted (e.g., no S2L1C)")
    print("  - Metrics are shown as: mean ± std across seeds")
    print("  - Format: CopGen / TerraMind (left/right of slash)")
    print("  - '-' indicates missing data for that configuration")

    models = agg["model"].unique().tolist()
    if models:
        print(f"\nModels included: {' / '.join(models)}")
    print("=" * 80)

    if agg.empty:
        print("No aggregated results found.")
        return
    # Sort stable order of targets
    targets = [t for t in ["DEM", "LULC", "S1RTC", "S2L1C", "S2L2A", "lat_lon"] if t in agg["target"].unique()]
    for target in targets:
        metrics_to_show = METRICS_FOR_TARGET_TABLE[target]
        target_rows = agg[agg["target"] == target]
        if target_rows.empty:
            continue
        condition_keys = sorted(target_rows["condition_key"].unique(), key=_condition_sort_key)
        # Header
        print(f"\n### Target Modality: {target}\n")
        col_titles = "Variant         | Condition (Input)                        | n_samples         | " + " | ".join([f"{m}" for m in metrics_to_show]) + " |"
        sep_line = "-" * max(len(col_titles), 120)
        print(sep_line)
        print(col_titles)
        print(sep_line)

        for condition_key in condition_keys:
            rows_by_model: Dict[str, Optional[pd.Series]] = {
                "copgen": _find_row(agg, "copgen", target, condition_key),
                "terramind": _find_row(agg, "terramind", target, condition_key),
            }
            metric_cells: List[str] = []
            cop_row = rows_by_model["copgen"]
            ter_row = rows_by_model["terramind"]
            cop_n = str(int(cop_row["n_samples"])) if cop_row is not None and "n_samples" in cop_row else "-"
            ter_n = str(int(ter_row["n_samples"])) if ter_row is not None and "n_samples" in ter_row else "-"
            ns_cell = f"{cop_n} / {ter_n}"
            for metric in metrics_to_show:
                cop_cell = "-"
                ter_cell = "-"
                if cop_row is not None:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    cop_cell = _format_mean_std(float(cop_row.get(mean_col, np.nan)), float(cop_row.get(std_col, np.nan)), float_fmt=float_fmt)
                if ter_row is not None:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    ter_cell = _format_mean_std(float(ter_row.get(mean_col, np.nan)), float(ter_row.get(std_col, np.nan)), float_fmt=float_fmt)
                metric_cells.append(f"{cop_cell} / {ter_cell}")

            missing_extra = _missing_from_condition_key(condition_key)
            variant_label = _variant_label(condition_key)
            cond_str = _condition_string(target, missing_extra)
            print(f"{variant_label:<15} | {cond_str:<36} | {ns_cell:<17} | " + " | ".join(f"{c:<23}" for c in metric_cells) + " |")
        print("=" * 80)


def print_tile_mean_tables(tile_agg: pd.DataFrame, float_fmt: str) -> None:
    print("\n" + "=" * 80)
    print("\033[31m=== TILE-LEVEL MEAN ± STD RESULTS ===\033[0m")
    print("PER-TILE CROSS-MODAL RESULTS (mean ± std across all tiles)")
    print("=" * 80)
    print("\nHow to read the tables:")
    print("  - For each tile we first average metrics across seeds, then compute dataset-wide mean ± std")
    print("  - No best-of selection: every tile contributes after the per-tile averaging step")
    print("  - Column 'n_tiles (runs)' shows the number of tile samples and unique seeds")
    print("  - Format: CopGen / TerraMind (left/right of slash)")
    print("  - '-' indicates missing data for that configuration")

    if tile_agg.empty:
        print("No per-tile aggregated results found.")
        return

    targets = [t for t in ["DEM", "LULC", "S1RTC", "S2L1C", "S2L2A", "lat_lon"] if t in tile_agg["target"].unique()]
    for target in targets:
        metrics_to_show = METRICS_FOR_TARGET_TABLE[target]
        target_rows = tile_agg[tile_agg["target"] == target]
        if target_rows.empty:
            continue
        condition_keys = sorted(target_rows["condition_key"].unique(), key=_condition_sort_key)
        print(f"\n### Target Modality: {target} (tile-level mean)\n")
        col_titles = "Variant         | Condition (Input)                        | n_tiles (runs)    | " + " | ".join([f"{m}" for m in metrics_to_show]) + " |"
        sep_line = "-" * max(len(col_titles), 120)
        print(sep_line)
        print(col_titles)
        print(sep_line)

        for condition_key in condition_keys:
            rows_by_model: Dict[str, Optional[pd.Series]] = {
                "copgen": _find_row(tile_agg, "copgen", target, condition_key, count_col="n_tiles"),
                "terramind": _find_row(tile_agg, "terramind", target, condition_key, count_col="n_tiles"),
            }
            metric_cells: List[str] = []
            cop_row = rows_by_model["copgen"]
            ter_row = rows_by_model["terramind"]
            cop_tiles = str(int(cop_row["n_tiles"])) if cop_row is not None and "n_tiles" in cop_row else "-"
            ter_tiles = str(int(ter_row["n_tiles"])) if ter_row is not None and "n_tiles" in ter_row else "-"
            cop_runs = str(int(cop_row["n_runs"])) if cop_row is not None and "n_runs" in cop_row else "-"
            ter_runs = str(int(ter_row["n_runs"])) if ter_row is not None and "n_runs" in ter_row else "-"
            cop_counts = cop_tiles if cop_runs == "-" else f"{cop_tiles} ({cop_runs})"
            ter_counts = ter_tiles if ter_runs == "-" else f"{ter_tiles} ({ter_runs})"
            counts_cell = f"{cop_counts} / {ter_counts}"

            for metric in metrics_to_show:
                cop_cell = "-"
                ter_cell = "-"
                if cop_row is not None:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    cop_cell = _format_mean_std(float(cop_row.get(mean_col, np.nan)), float(cop_row.get(std_col, np.nan)), float_fmt=float_fmt)
                if ter_row is not None:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    ter_cell = _format_mean_std(float(ter_row.get(mean_col, np.nan)), float(ter_row.get(std_col, np.nan)), float_fmt=float_fmt)
                metric_cells.append(f"{cop_cell} / {ter_cell}")

            missing_extra = _missing_from_condition_key(condition_key)
            variant_label = _variant_label(condition_key)
            cond_str = _condition_string(target, missing_extra)
            print(f"{variant_label:<15} | {cond_str:<36} | {counts_cell:<17} | " + " | ".join(f"{c:<23}" for c in metric_cells) + " |")
        print("=" * 80)


def _format_single_value(val: Optional[float], float_fmt: str) -> str:
    if val is None or (isinstance(val, float) and (np.isnan(val))):
        return "-"
    return f"{val:{float_fmt}}"


def _select_rows(df: pd.DataFrame, model: str, target: str, condition_key: str) -> pd.DataFrame:
    return df[
        (df["model"] == model)
        & (df["target"] == target)
        & (df["condition_key"] == condition_key)
    ]


def print_best_comparison_tables(best_df: pd.DataFrame, float_fmt: str, source: str) -> None:
    print("\n" + "=" * 80)
    print("\033[31m=== TILE-LEVEL-TAKE-BEST RESULTS ===\033[0m")
    print("BEST-OF COMPARISON (tile-level best sample per configuration)")
    print(f"Source: {source}")
    print("=" * 80)
    if best_df.empty:
        print("No best-of results found.")
        return
    targets = [t for t in ["DEM", "LULC", "S1RTC", "S2L1C", "S2L2A", "lat_lon"] if t in best_df["target"].unique()]
    for target in targets:
        metrics_to_show = METRICS_FOR_TARGET_TABLE[target]
        target_rows = best_df[best_df["target"] == target]
        if target_rows.empty:
            continue
        condition_keys = sorted(target_rows["condition_key"].unique(), key=_condition_sort_key)
        print(f"\n### Target Modality: {target} (best across seeds)\n")
        col_titles = "Variant         | Condition (Input)                        | n_runs            | " + " | ".join([f"{m}" for m in metrics_to_show]) + " |"
        sep_line = "-" * max(len(col_titles), 120)
        print(sep_line)
        print(col_titles)
        print(sep_line)
        for condition_key in condition_keys:
            rows_by_model: Dict[str, pd.DataFrame] = {
                "copgen": _select_rows(best_df, "copgen", target, condition_key),
                "terramind": _select_rows(best_df, "terramind", target, condition_key),
            }
            metric_cells: List[str] = []
            cop_n = "-"
            ter_n = "-"
            if not rows_by_model["copgen"].empty and "n_runs" in rows_by_model["copgen"].columns:
                cop_n = str(int(rows_by_model["copgen"]["n_runs"].max()))
            if not rows_by_model["terramind"].empty and "n_runs" in rows_by_model["terramind"].columns:
                ter_n = str(int(rows_by_model["terramind"]["n_runs"].max()))
            ns_cell = f"{cop_n} / {ter_n}"
            for metric in metrics_to_show:
                higher_better = (metric in HIGHER_BETTER_METRICS) or (metric not in LOWER_BETTER_METRICS)
                cop_best = None
                ter_best = None
                df_cop = rows_by_model["copgen"]
                if not df_cop.empty and metric in df_cop.columns:
                    series = pd.to_numeric(df_cop[metric], errors="coerce")
                    if series.notna().any():
                        cop_best = float(series.max()) if higher_better else float(series.min())
                df_ter = rows_by_model["terramind"]
                if not df_ter.empty and metric in df_ter.columns:
                    series = pd.to_numeric(df_ter[metric], errors="coerce")
                    if series.notna().any():
                        ter_best = float(series.max()) if higher_better else float(series.min())
                cop_cell = _format_single_value(cop_best, float_fmt)
                ter_cell = _format_single_value(ter_best, float_fmt)
                metric_cells.append(f"{cop_cell} / {ter_cell}")
            missing_extra = _missing_from_condition_key(condition_key)
            variant_label = _variant_label(condition_key)
            cond_str = _condition_string(target, missing_extra)
            print(f"{variant_label:<15} | {cond_str:<36} | {ns_cell:<17} | " + " | ".join(f"{c:<23}" for c in metric_cells) + " |")
        print("=" * 80)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarize best-of-n cross-modal metrics for COP-GEN and TerraMind")
    parser.add_argument("--dataset_root", "-d", required=True, type=str, help="Dataset root containing outputs/{copgen,terramind}")
    parser.add_argument("--export_csv", type=str, default=None, help="Optional path to save aggregated CSV")
    parser.add_argument("--precision", type=int, default=2, help="Number of decimal places for printed numbers (default: 2)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    raw_df = scan_all_runs(dataset_root)
    tile_df = scan_all_runs_tiles(dataset_root)
    if raw_df.empty:
        print("No runs found under", dataset_root / "outputs")
        return
    agg_df = aggregate_over_seeds(raw_df)
    best_tile_df = aggregate_best_per_tile(tile_df)
    tile_mean_std_df = aggregate_tile_mean_std(tile_df)
    if args.export_csv:
        Path(args.export_csv).parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(args.export_csv, index=False)
    float_fmt = f".{max(0, args.precision)}f"
    print_summary_tables(agg_df, float_fmt)
    if tile_mean_std_df.empty:
        print("\nNo per-tile metrics found for mean/std aggregation.")
    else:
        print_tile_mean_tables(tile_mean_std_df, float_fmt)
    if best_tile_df.empty:
        print("\nNo per-tile metrics found; falling back to experiment-level best-of (output_metrics.txt).")
        # print_best_comparison_tables(raw_df, float_fmt, source="experiment-level output_metrics.txt")
    else:
        print_best_comparison_tables(best_tile_df, float_fmt, source="per-tile best (output_metrics_per_tile.csv)")


if __name__ == "__main__":
    main()

