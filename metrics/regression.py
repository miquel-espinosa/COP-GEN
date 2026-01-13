import torch
import numpy as np
from typing import Dict
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.aggregation import CatMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

EARTH_RADIUS_KM = 6371.0

def _summarize_tensor(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {k: 0.0 for k in ['mean', 'std', 'rmse', 'p50', 'p90', 'p95', 'p99']}
    v = values.float().view(-1)
    mean = torch.mean(v)
    # match population std used previously
    std = torch.std(v, unbiased=False)
    rmse = torch.sqrt(torch.mean(v ** 2))
    q = torch.tensor([0.5, 0.9, 0.95, 0.99], device=v.device, dtype=v.dtype)
    p50, p90, p95, p99 = torch.quantile(v, q).tolist()
    return {
        'mean': float(mean.item()),
        'std': float(std.item()),
        'rmse': float(rmse.item()),
        'p50': float(p50),
        'p90': float(p90),
        'p95': float(p95),
        'p99': float(p99),
    }

class ImageRegressionMetrics:
    def __init__(self):
        self.mae_metric = MeanAbsoluteError()
        self.rmse_metric = MeanSquaredError(squared=False)
        # SSIM expects inputs in [0, 1]; we'll normalize per-sample before update
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0,1))
        # PSNR also expects known data range; use normalized [0,1]
        self.psnr_metric = PeakSignalNoiseRatio(data_range=(0,1))
        self._device = None

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred/target: [B, C, H, W]
        p = pred.detach().float()
        t = target.detach().float()
        # ensure metrics live on same device as inputs
        if self._device != p.device:
            self.mae_metric = self.mae_metric.to(p.device)
            self.rmse_metric = self.rmse_metric.to(p.device)
            self.ssim_metric = self.ssim_metric.to(p.device)
            self.psnr_metric = self.psnr_metric.to(p.device)
            self._device = p.device
        # Update MAE/RMSE in original scale
        self.mae_metric.update(p, t)
        self.rmse_metric.update(p, t)
        # Normalize each sample to [0,1] using target min/max and update SSIM
        # Avoid division by zero by adding eps; clamp to [0,1]
        B = t.shape[0]
        t_flat = t.view(B, -1)
        t_min = t_flat.min(dim=1).values.view(B, 1, 1, 1)
        t_max = t_flat.max(dim=1).values.view(B, 1, 1, 1)
        denom = (t_max - t_min).clamp_min(1e-6)
        t01 = ((t - t_min) / denom).clamp(0.0, 1.0)
        p01 = ((p - t_min) / denom).clamp(0.0, 1.0)
        self.ssim_metric.update(p01, t01)
        self.psnr_metric.update(p01, t01)

    def summary(self) -> Dict[str, float]:
        mae = float(self.mae_metric.compute().item())
        rmse = float(self.rmse_metric.compute().item())
        ssim = float(self.ssim_metric.compute().item())
        psnr = float(self.psnr_metric.compute().item())
        return {
            'mae': mae,
            'rmse': rmse,
            'ssim': ssim,
            'psnr': psnr,
        }

# Spatial metrics: lat/lon errors in km using haversine distance
class SpatialMetrics:
    def __init__(self, radii_km=(1, 10, 100, 500)):
        self.radii = radii_km
        self.distances = CatMetric()
        self.total = 0
        self.hits = {r: 0 for r in self.radii}
        self._device = None

    @staticmethod
    def _haversine_km_torch(latlon_p: torch.Tensor, latlon_t: torch.Tensor) -> torch.Tensor:
        # latlon_*: [..., 2] in degrees
        lat1 = torch.deg2rad(latlon_p[..., 0].float())
        lon1 = torch.deg2rad(latlon_p[..., 1].float())
        lat2 = torch.deg2rad(latlon_t[..., 0].float())
        lon2 = torch.deg2rad(latlon_t[..., 1].float())
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
        c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))
        return torch.tensor(EARTH_RADIUS_KM, dtype=c.dtype, device=c.device) * c

    @torch.no_grad()
    def update(self, pred_latlon: torch.Tensor, target_latlon: torch.Tensor):
        # shape [B, 2] or [B, 1, 1, 2]
        p = pred_latlon.reshape(-1, 2).detach()
        t = target_latlon.reshape(-1, 2).detach()
        d = self._haversine_km_torch(p, t).view(-1)
        if self._device != d.device:
            self.distances = self.distances.to(d.device)
            self._device = d.device
        self.total += int(d.numel())
        self.distances.update(d)
        for r in self.radii:
            self.hits[r] += int((d <= r).sum().item())

    def summary(self) -> Dict[str, float]:
        s = _summarize_tensor(self.distances.compute())
        acc = {f'acc@{r}km': (self.hits[r] / self.total if self.total > 0 else 0.0) for r in self.radii}
        return {
            'median_km': s['p50'],
            'mean_km': s['mean'],
            'std_km': s['std'],
            'rmse_km': s['rmse'],
            'p90_km': s['p90'],
            'p95_km': s['p95'],
            'p99_km': s['p99'],
            **acc,
        }

# Temporal metrics: timestamp error in days
class TemporalMetrics:
    def __init__(self):
        self.abs_days = CatMetric()
        # Track absolute month difference in [0-11]
        self.abs_months = CatMetric()
        # Track correctness for month and year as accuracies
        self.month_correct = 0
        self.month_total = 0
        self.year_correct = 0
        self.year_total = 0
        self._device = None

    @torch.no_grad()
    def update(self, pred_date_ymd: torch.Tensor, target_date_ymd: torch.Tensor):
        # tensors of shape [B, 3] with [day, month, year]
        p = pred_date_ymd.detach().cpu().numpy()
        t = target_date_ymd.detach().cpu().numpy()
        from datetime import datetime
        diffs = []
        month_diffs = []
        for i in range(p.shape[0]):
            pd, pm, py = int(p[i,0]), int(p[i,1]), int(p[i,2])
            td, tm, ty = int(t[i,0]), int(t[i,1]), int(t[i,2])
            try:
                dp = datetime(py, pm, pd)
                dt = datetime(ty, tm, td)
                diffs.append(abs((dp - dt).days))
                # Month absolute difference (ignoring year)
                month_diffs.append(abs(pm - tm))
                # Month accuracy (exact match)
                self.month_correct += int(pm == tm)
                self.month_total += 1
                # Year accuracy (exact match)
                self.year_correct += int(py == ty)
                self.year_total += 1
            except Exception:
                raise ValueError(f"Invalid date: {pd}-{pm}-{py} or {td}-{tm}-{ty}")
        if diffs:
            diffs_t = torch.tensor(diffs, dtype=torch.float32)
            if self._device != diffs_t.device:
                self.abs_days = self.abs_days.to(diffs_t.device)
                self.abs_months = self.abs_months.to(diffs_t.device)
                self._device = diffs_t.device
            self.abs_days.update(diffs_t)
        if month_diffs:
            m_t = torch.tensor(month_diffs, dtype=torch.float32, device=self._device or 'cpu')
            # ensure metrics devices are aligned
            if self._device != m_t.device:
                self.abs_months = self.abs_months.to(m_t.device)
                self._device = m_t.device
            self.abs_months.update(m_t)

    def summary(self) -> Dict[str, float]:
        s = _summarize_tensor(self.abs_days.compute())
        sm = _summarize_tensor(self.abs_months.compute())
        month_acc = (self.month_correct / self.month_total) if self.month_total > 0 else 0.0
        year_acc = (self.year_correct / self.year_total) if self.year_total > 0 else 0.0
        return {
            'median_days': s['p50'],
            'mean_days': s['mean'],
            'std_days': s['std'],
            'rmse_days': s['rmse'],
            'p90_days': s['p90'],
            'p95_days': s['p95'],
            'p99_days': s['p99'],
            # Month-specific metrics
            'mean_abs_months': sm['mean'],
            'median_abs_months': sm['p50'],
            'p90_abs_months': sm['p90'],
            'month_acc': month_acc,
            # Year-specific metric
            'year_acc': year_acc,
        }