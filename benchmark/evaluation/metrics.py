from __future__ import annotations

from typing import Optional
from .regression import ImageRegressionMetrics, SpatialMetrics
from .classification import SegmentationMetrics


def select_metrics(output_modality: str):
    if output_modality == "coords":
        return SpatialMetrics(radii_km=(1, 10, 50, 100, 500))
    if output_modality == "LULC":
        return SegmentationMetrics(num_classes=10, topk=3)
    return ImageRegressionMetrics()


