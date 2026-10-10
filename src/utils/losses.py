"""
损失函数 — 源自 LitePT/models/losses/
包含: Lovasz Loss, CrossEntropyLoss, FocalLoss, DiceLoss
"""
from typing import Optional
from itertools import filterfalse
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Lovasz Loss
# ============================================================

def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    return scores[valid], labels[valid]


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    for c in labels.unique():
        fg = (labels == c).type_as(probas)
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(
                *_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes,
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(
            *_flatten_probas(probas, labels, ignore),
            classes=classes,
        )
    return loss


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):
    """Lovasz Loss for segmentation"""

    def __init__(
        self,
        mode: str = "multiclass",
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        assert mode in {"binary", "multiclass", "multilabel"}
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        if self.mode in {"binary", "multilabel"}:
            loss = _lovasz_hinge(
                y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index
            )
        elif self.mode == "multiclass":
            y_pred = y_pred.softmax(dim=1)
            loss = _lovasz_softmax(
                y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index
            )
        else:
            raise ValueError(f"Wrong mode {self.mode}")
        return loss * self.loss_weight


# ============================================================
# 其他常用损失
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss"""

    def __init__(self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.transpose(0, 1).reshape(pred.size(0), -1).transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device)

        num_classes = pred.size(1)
        target_oh = F.one_hot(target, num_classes=num_classes)
        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target_t = target_oh.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target_t + pred_sigmoid * (1 - target_t)
        focal_weight = (alpha * target_t + (1 - alpha) * (1 - target_t)) * one_minus_pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target_t, reduction="none") * focal_weight
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.transpose(0, 1).reshape(pred.size(0), -1).transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_oh = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)

        total_loss = 0
        for i in range(num_classes):
            num = torch.sum(torch.mul(pred[:, i], target_oh[:, i].float())) * 2 + self.smooth
            den = torch.sum(pred[:, i].pow(self.exponent) + target_oh[:, i].pow(self.exponent).float()) + self.smooth
            total_loss += 1 - num / den
        return self.loss_weight * total_loss / num_classes
