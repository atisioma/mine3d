"""
DataLoader 构建 — 处理变长点云的 padding + collate
"""
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import IndoorSegDataset


def collate_fn(batch):
    """
    自定义 collate: 对变长点云做 padding 到 batch 内最大长度
    输入: list of dict {points(N,3), normals(N,3), colors(N,3), labels(N)}
    输出: padded tensors
    """
    max_n = max(s["points"].shape[0] for s in batch)
    B = len(batch)

    # 拼接 xyz + rgb -> (B, N, 6)
    inputs = torch.zeros(B, max_n, 6)
    labels = torch.zeros(B, max_n, dtype=torch.long)
    masks = torch.zeros(B, max_n, dtype=torch.bool)  # True = 有效点

    for i, s in enumerate(batch):
        n = s["points"].shape[0]
        feat = np.concatenate([s["points"], s["colors"]], axis=1)  # (N, 6)
        inputs[i, :n] = torch.from_numpy(feat)
        labels[i, :n] = torch.from_numpy(s["labels"])
        masks[i, :n] = True

    return {"inputs": inputs, "labels": labels, "masks": masks}


def build_dataloader(data_dir, split="train", batch_size=4, num_workers=0):
    ds = IndoorSegDataset(data_dir, split=split)
    shuffle = (split == "train")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )
    return loader


if __name__ == "__main__":
    loader = build_dataloader("data", split="train", batch_size=2)
    batch = next(iter(loader))
    print("inputs:", batch["inputs"].shape, batch["inputs"].dtype)
    print("labels:", batch["labels"].shape, batch["labels"].dtype)
    print("masks: ", batch["masks"].shape, batch["masks"].dtype)
    print("有效点数:", batch["masks"].sum(dim=1).tolist())
