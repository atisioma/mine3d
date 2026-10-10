"""
语义分割评估指标 — 源自 LitePT/metrics/semantic.py
简化版: 不依赖 torchmetrics, 纯 torch 实现
"""
import os
import torch
import numpy as np


class ConfusionMatrix:
    """混淆矩阵 — 计算 OA, mAcc, mIoU, per-class IoU"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self):
        self.confmat.zero_()

    @torch.no_grad()
    def update(self, pred, target):
        """
        pred:   (N,) or (N, C) logits — 如果是 logits 会取 argmax
        target: (N,) labels
        """
        assert pred.shape[0] == target.shape[0]

        if pred.dim() == 2:
            pred = pred.argmax(dim=1)
        if pred.is_floating_point():
            pred = pred.long()
        target = target.long()

        # 过滤无效标签
        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        # 累加混淆矩阵
        indices = target * self.num_classes + pred
        cm = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confmat += cm.reshape(self.num_classes, self.num_classes).long()

    def iou(self, as_percent=True):
        TP_plus_FN = self.confmat.sum(dim=0)
        TP_plus_FP = self.confmat.sum(dim=1)
        TP = self.confmat.diag()
        union = TP_plus_FN + TP_plus_FP - TP
        iou = 1e-8 + TP.float() / (union.float() + 1e-8)
        existing_class_mask = union > 1e-3
        if as_percent:
            iou *= 100
        return iou, existing_class_mask

    def oa(self, as_percent=True):
        correct = self.confmat.diag().sum().item()
        total = self.confmat.sum().item()
        if total == 0:
            return 0
        if as_percent:
            correct *= 100
        return correct / total

    def miou(self, missing_as_one=False, as_percent=True):
        values, mask = self.iou(as_percent=as_percent)
        if mask.sum() == 0:
            return 0
        if missing_as_one:
            values[~mask] = 1
            mask[:] = True
        return values[mask].sum() / mask.sum()

    def macc(self, as_percent=True):
        re = 0
        label_presents = 0
        for i in range(self.num_classes):
            total_gt = self.confmat[i, :].sum().item()
            if total_gt:
                label_presents += 1
                re += self.confmat[i, i].item() / total_gt
        if label_presents == 0:
            return 0
        if as_percent:
            re *= 100
        return re / label_presents

    def all_metrics(self, as_percent=True):
        return {
            "oa": self.oa(as_percent),
            "macc": self.macc(as_percent),
            "miou": self.miou(as_percent),
            "iou_per_class": self.iou(as_percent)[0],
        }

    def print_metrics(self, class_names=None):
        oa = self.oa()
        macc = self.macc()
        miou = self.miou()
        iou_vals, seen = self.iou()

        print(f"OA:   {oa:0.2f}")
        print(f"mAcc: {macc:0.2f}")
        print(f"mIoU: {miou:0.2f}")

        names = class_names or [f"class-{i}" for i in range(self.num_classes)]
        for c, iou_v, s in zip(names, iou_vals, seen):
            if not s:
                print(f"  {c:<13}: not seen")
            else:
                print(f"  {c:<13}: {iou_v:0.2f}")
