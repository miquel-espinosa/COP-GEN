"""
Embedding extractors and perceptual metrics for the COP-GEN benchmark.

Two DNN-based components:
  1. ResNet50 (ImageNet) — RGB embeddings for Stream 1 perceptual metrics
  2. LPIPS — learned perceptual distance for intra-set diversity

Both operate on RGB (B04/B03/B02) and are agnostic to L1C/L2A processing
level, avoiding the mismatch issue with SSL4EO-S12 (trained on L1C).
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

log = logging.getLogger(__name__)

# Band indices in the 12-band stack for RGB
RGB_BANDS = [3, 2, 1]  # B04 (Red), B03 (Green), B02 (Blue)


def _to_rgb_tensor(imgs: np.ndarray) -> torch.Tensor:
    """Convert (N, 12, H, W) float32 → (N, 3, H, W) clamped to [0, 1]."""
    t = torch.from_numpy(imgs[:, RGB_BANDS]).float()
    return t.clamp(0, 1)


# ---------------------------------------------------------------------------
# ResNet50 (ImageNet) embeddings
# ---------------------------------------------------------------------------

class ResNet50Embedder:
    """ResNet50 (ImageNet) RGB embedding extractor.

    Extracts 2048-D feature vectors from the global average pool layer.
    Input: B04/B03/B02 normalised to [0, 1], resized to 224×224.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        log.info("ResNet50 (ImageNet) loaded on %s", self.device)

    @torch.no_grad()
    def embed(self, imgs: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Extract embeddings from (N, 12, 192, 192) images.

        Returns (N, 2048) float32 numpy array.
        """
        rgb = _to_rgb_tensor(imgs)

        embeddings = []
        for i in range(0, len(rgb), batch_size):
            batch = rgb[i:i + batch_size].to(self.device)
            batch = nn.functional.interpolate(batch, size=224, mode="bilinear", align_corners=False)
            batch = torch.stack([self.normalize(b) for b in batch])
            feat = self.model(batch).squeeze(-1).squeeze(-1)
            embeddings.append(feat.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# LPIPS intra-set distance
# ---------------------------------------------------------------------------

class LPIPSDistance:
    """LPIPS (Learned Perceptual Image Patch Similarity) for measuring
    perceptual diversity within a set of images.

    Uses AlexNet backbone (fastest, standard choice).
    """

    def __init__(self, device: str = "cuda"):
        import lpips
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net="alex", verbose=False).to(self.device)
        self.model.eval()
        log.info("LPIPS (AlexNet) loaded on %s", self.device)

    @torch.no_grad()
    def intra_set_distance(self, imgs: np.ndarray, max_pairs: int = 500) -> float:
        """Mean pairwise LPIPS distance within a set.

        Parameters
        ----------
        imgs : (N, 12, H, W) float32, normalised [0, 1]
        max_pairs : int
            Cap on number of pairs to evaluate (for speed with large N).
            With N=33 (COP-GEN), there are 528 pairs — we cap at 500.

        Returns
        -------
        mean LPIPS distance (float), or nan if N < 2.
        """
        n = len(imgs)
        if n < 2:
            return float("nan")

        # LPIPS expects (N, 3, H, W) in [-1, 1]
        rgb = _to_rgb_tensor(imgs)  # (N, 3, H, W) in [0, 1]
        rgb = rgb * 2 - 1  # scale to [-1, 1]

        pairs = list(combinations(range(n), 2))
        if len(pairs) > max_pairs:
            import random
            random.seed(42)
            pairs = random.sample(pairs, max_pairs)

        distances = []
        # Process in batches of pairs
        batch_size = 32
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            imgs_a = torch.stack([rgb[i] for i, _ in batch_pairs]).to(self.device)
            imgs_b = torch.stack([rgb[j] for _, j in batch_pairs]).to(self.device)
            d = self.model(imgs_a, imgs_b).squeeze()
            if d.dim() == 0:
                distances.append(d.item())
            else:
                distances.extend(d.cpu().tolist())

        return float(np.mean(distances))
