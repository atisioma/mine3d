"""
PLY 数据集加载器
标准格式: x y z nx ny nz red green blue class
"""
import numpy as np
from pathlib import Path


def read_ply(filepath):
    """
    读取标准 PLY 文件 (ASCII) — 向量化快速解析
    返回: points(N,3), normals(N,3), colors(N,3), labels(N)
    """
    n_points = 0
    header_end = 0
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                n_points = int(stripped.split()[-1])
            if stripped == "end_header":
                header_end = i + 1
                break

    # 一次性读取所有数据行
    data_lines = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i >= header_end:
                data_lines.append(line)
                if len(data_lines) >= n_points:
                    break

    # 用 numpy 向量化解析
    data = np.array([list(map(float, line.split())) for line in data_lines])

    points = data[:, 0:3].astype(np.float32)
    normals = data[:, 3:6].astype(np.float32)
    colors = data[:, 6:9].astype(np.uint8)
    labels = data[:, 9].astype(np.int32)

    return points, normals, colors, labels


class IndoorSegDataset:
    """室内语义分割数据集"""

    CLASSES = {0: "table", 1: "apple", 2: "banana"}
    NUM_CLASSES = 3

    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # 标准目录结构: data_dir/train/ 和 data_dir/val/
        split_dir = self.data_dir / split
        self.files = sorted(split_dir.glob("*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        points, normals, colors, labels = read_ply(str(filepath))

        # 归一化坐标到 [-1, 1]
        points = points - points.mean(axis=0)
        max_val = np.abs(points).max()
        if max_val > 0:
            points = points / max_val

        # 颜色归一化到 [0, 1]
        colors = colors.astype(np.float32) / 255.0

        sample = {
            "points": points,       # (N, 3) float32
            "normals": normals,     # (N, 3) float32
            "colors": colors,       # (N, 3) float32
            "labels": labels,       # (N,)   int32
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    ds = IndoorSegDataset("data", split="train")
    print(f"训练样本数: {len(ds)}")
    sample = ds[0]
    print(f"points:    {sample['points'].shape} {sample['points'].dtype}")
    print(f"normals:   {sample['normals'].shape} {sample['normals'].dtype}")
    print(f"colors:    {sample['colors'].shape} {sample['colors'].dtype}")
    print(f"labels:    {sample['labels'].shape} {sample['labels'].dtype}")
    print(f"类别分布:  {dict(zip(*np.unique(sample['labels'], return_counts=True)))}")
