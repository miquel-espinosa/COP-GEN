from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank COP-GEN samples for a tile by metrics stored in metrics/sample_*.json",
        epilog="Example: python rank_samples.py --tile-root /path/to/tile_root --topk 5",
    )
    p.add_argument("--tile-root", type=str, required=True, help="Path to tile root (contains metrics/, samples/, etc.)")
    p.add_argument("--topk", type=int, default=5, help="Number of top-k samples to print per metric and overall")
    return p.parse_args()


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # filter NaN if any


def determine_higher_is_better(metric_name: str, modality: str) -> bool:
    """
    Heuristic to decide whether higher values are better for a given metric name.
    Covers regression, segmentation, spatial and temporal metric names used in the repo.
    """
    name = metric_name.lower()
    # Positive metrics
    positive_tokens = ("ssim", "psnr", "acc@", "iou", "f1")
    if any(tok in name for tok in positive_tokens):
        return True
    if name.startswith("overall_top"):
        return True
    if name.endswith("_acc") or name in ("month_acc", "year_acc"):
        return True

    # Negative metrics (errors / deviations)
    negative_tokens = (
        "mae",
        "rmse",
        "median",
        "p90",
        "p95",
        "p99",
        "_km",
        "_days",
        "abs_months",
        "std_km",
        "std_days",
    )
    if any(tok in name for tok in negative_tokens):
        return False

    # Defaults:
    # - For known image regression modalities, default to minimizing (conservative for unknown errors)
    # - For known classification modalities, default to maximizing
    if modality in ("LULC", "cloud_mask"):
        return True
    return False


def load_sample_metrics(metrics_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load all sample_*.json under metrics_dir into:
      { sample_id: { modality: { metric_name: value, ... }, ... }, ... }
    Only keep scalar numeric metrics; ignore dict/list fields like 'per_band' or per-class arrays.
    """
    by_sample: Dict[str, Dict[str, Dict[str, float]]] = {}
    for jf in sorted(metrics_dir.glob("sample_*.json")):
        sample_id = jf.stem  # e.g., "sample_0"
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        sample_entry: Dict[str, Dict[str, float]] = {}
        for modality, md in data.items():
            if not isinstance(md, dict):
                continue
            scalars: Dict[str, float] = {}
            for k, v in md.items():
                if is_number(v):
                    scalars[k] = float(v)
                # skip lists/dicts (e.g., per_band, per_class arrays)
            if scalars:
                sample_entry[modality] = scalars
        if sample_entry:
            by_sample[sample_id] = sample_entry
    return by_sample


def collect_per_metric_series(by_sample: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Transform into:
      { modality: { metric_name: [(sample_id, value), ...], ... }, ... }
    """
    per_mod_metric: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for sample_id, mod_map in by_sample.items():
        for modality, metrics in mod_map.items():
            for metric_name, value in metrics.items():
                per_mod_metric.setdefault(modality, {}).setdefault(metric_name, []).append((sample_id, value))
    return per_mod_metric


def topk_for_metric(
    series: List[Tuple[str, float]],
    higher_is_better: bool,
    k: int,
) -> List[Tuple[str, float]]:
    if not series:
        return []
    reverse = higher_is_better
    sorted_series = sorted(series, key=lambda x: x[1], reverse=reverse)
    return sorted_series[: max(0, min(k, len(sorted_series)))]


def compute_overall_ranking(
    per_mod_metric: Dict[str, Dict[str, List[Tuple[str, float]]]]
) -> List[Tuple[str, float, int]]:
    """
    Compute overall ranking by averaging ranks across all metrics where a sample appears.
    For each metric, we compute ranks (1 = best) with direction given by determine_higher_is_better.
    Returns list of (sample_id, avg_rank, count_metrics_contributed), sorted by avg_rank ascending.
    """
    # Collect all sample ids
    all_samples: set[str] = set()
    for _, md in per_mod_metric.items():
        for _, series in md.items():
            all_samples.update(s for s, _ in series)

    # Accumulate ranks per sample
    rank_sums: Dict[str, float] = {s: 0.0 for s in all_samples}
    rank_counts: Dict[str, int] = {s: 0 for s in all_samples}

    for modality, metrics in per_mod_metric.items():
        for metric_name, series in metrics.items():
            if not series:
                continue
            higher_is_better = determine_higher_is_better(metric_name, modality)
            # Sort per direction
            series_sorted = sorted(series, key=lambda x: x[1], reverse=higher_is_better)
            # Assign ranks starting at 1 (ties not specially handled)
            for rank_idx, (sample_id, _) in enumerate(series_sorted, start=1):
                rank_sums[sample_id] += float(rank_idx)
                rank_counts[sample_id] += 1

    scored: List[Tuple[str, float, int]] = []
    for s in all_samples:
        cnt = rank_counts.get(s, 0)
        if cnt > 0:
            avg_rank = rank_sums[s] / float(cnt)
            scored.append((s, avg_rank, cnt))
    scored.sort(key=lambda x: x[1])
    return scored


def pretty_print_per_metric(per_mod_metric: Dict[str, Dict[str, List[Tuple[str, float]]]], topk: int) -> None:
    print("=" * 60)
    print("Per-modality and per-metric top-k samples")
    print("=" * 60)
    for modality in sorted(per_mod_metric.keys()):
        print(f"\n[{modality}]")
        metrics = per_mod_metric[modality]
        for metric_name in sorted(metrics.keys()):
            higher_is_better = determine_higher_is_better(metric_name, modality)
            series = metrics[metric_name]
            top = topk_for_metric(series, higher_is_better, topk)
            direction = "desc" if higher_is_better else "asc"
            print(f"  - {metric_name} (sorted {direction}):")
            for idx, (sid, val) in enumerate(top, start=1):
                # Choose precision: 4 decimals by default
                print(f"      {idx}) {sid}: {val:.4f}")


def pretty_print_overall(scored: List[Tuple[str, float, int]], topk: int) -> None:
    print("\n" + "=" * 60)
    print("Overall best samples (average rank across all metrics)")
    print("=" * 60)
    top = scored[: max(0, min(topk, len(scored)))]
    for idx, (sid, avg_rank, cnt) in enumerate(top, start=1):
        print(f"  {idx}) {sid}: avg_rank={avg_rank:.3f} over {cnt} metrics")


def main() -> None:
    args = parse_args()
    tile_root = Path(args.tile_root)
    metrics_dir = tile_root / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics directory not found under tile root: {metrics_dir}")

    by_sample = load_sample_metrics(metrics_dir)
    if not by_sample:
        print(f"No sample_*.json found under {metrics_dir}")
        return

    per_mod_metric = collect_per_metric_series(by_sample)
    pretty_print_per_metric(per_mod_metric, args.topk)

    overall = compute_overall_ranking(per_mod_metric)
    pretty_print_overall(overall, args.topk)


if __name__ == "__main__":
    main()


