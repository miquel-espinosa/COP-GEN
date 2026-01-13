import torch
import numpy as np
from typing import Optional

class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, topk: int = 5):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.topk = topk
        # Confusion matrix: [C, C]
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        self.total = 0
        self.correct_top1 = 0
        self.correct_topk = 0

    @torch.no_grad()
    def update_from_logits(self, logits: torch.Tensor, target_onehot: torch.Tensor):
        """
        logits: [B, C, H, W]
        target_onehot: [B, C, H, W] one-hot or soft logits in [0,1]; will argmax to labels
        """
        assert logits.dim() == 4 and target_onehot.dim() == 4
        assert logits.shape[:1] == target_onehot.shape[:1] and logits.shape[2:] == target_onehot.shape[2:]
        # Convert GT to class indices
        target = target_onehot.argmax(dim=1)  # [B, H, W]
        pred = logits.argmax(dim=1)           # [B, H, W]
        # Respect ignore_index for counters
        if self.ignore_index is not None:
            keep_3d = target != self.ignore_index
        else:
            keep_3d = torch.ones_like(target, dtype=torch.bool)
        n = int(keep_3d.sum().item())
        self.total += n
        # Top-1 (pixel accuracy)
        self.correct_top1 += int(((pred == target) & keep_3d).sum().item())
        # Top-k
        if self.topk and self.topk > 1:
            topk_preds = logits.topk(k=min(self.topk, self.num_classes), dim=1).indices  # [B, k, H, W]
            target_exp = target.unsqueeze(1)  # [B, 1, H, W]
            in_topk = (topk_preds == target_exp).any(dim=1)  # [B, H, W]
            self.correct_topk += int((in_topk & keep_3d).sum().item())
        # Confusion matrix (always compute on CPU to avoid device mismatch)
        pred_flat = pred.view(-1)
        tgt_flat = target.view(-1)
        if self.ignore_index is not None:
            keep = tgt_flat != self.ignore_index
            pred_flat = pred_flat[keep]
            tgt_flat = tgt_flat[keep]
        pred_flat = pred_flat.to('cpu')
        tgt_flat = tgt_flat.to('cpu')
        cm = torch.bincount(self.num_classes * tgt_flat + pred_flat, minlength=self.num_classes**2)
        self.cm += cm.view(self.num_classes, self.num_classes)

    def per_class_iou(self):
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = tp + fp + fn + 1e-8
        return (tp / denom).cpu().numpy()

    def per_class_precision(self):
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        denom = tp + fp + 1e-8
        return (tp / denom).cpu().numpy()

    def per_class_recall(self):
        cm = self.cm.float()
        tp = cm.diag()
        fn = cm.sum(dim=1) - tp
        denom = tp + fn + 1e-8
        return (tp / denom).cpu().numpy()

    def mean_iou(self):
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = tp + fp + fn
        valid = denom > 0
        if valid.any():
            miou = (tp[valid] / (denom[valid] + 1e-8)).mean().item()
            return float(miou)
        return 0.0

    def per_class_f1(self):
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = (2 * tp) + fp + fn + 1e-8
        return (2 * tp / denom).cpu().numpy()

    def mean_f1(self):
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = (2 * tp) + fp + fn
        valid = denom > 0
        if valid.any():
            mf1 = ((2 * tp[valid]) / (denom[valid] + 1e-8)).mean().item()
            return float(mf1)
        return 0.0

    def frequency_weighted_iou(self):
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        gt_pixels_per_class = cm.sum(dim=1)
        denom = tp + fp + fn
        valid = denom > 0
        if valid.any():
            weighted_iou_sum = (gt_pixels_per_class[valid] * (tp[valid] / (denom[valid] + 1e-8))).sum()
            total_gt = gt_pixels_per_class[valid].sum() + 1e-8
            return float((weighted_iou_sum / total_gt).item())
        return 0.0

    def overall_top1(self):
        if self.total == 0:
            return 0.0
        return float(self.correct_top1) / float(self.total)

    def overall_topk(self):
        if self.total == 0 or not self.topk or self.topk <= 1:
            return 0.0
        return float(self.correct_topk) / float(self.total)

    def summary(self):
        return {
            'overall_top1': self.overall_top1(),
            'overall_topk': self.overall_topk(),
            'per_class_iou': self.per_class_iou().tolist(),
            'per_class_precision': self.per_class_precision().tolist(),
            'per_class_recall': self.per_class_recall().tolist(),
            'mean_iou': self.mean_iou(),
            'per_class_f1': self.per_class_f1().tolist(),
            'mean_f1': self.mean_f1(),
            'fw_iou': self.frequency_weighted_iou(),
        } 