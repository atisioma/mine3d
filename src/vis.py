"""
点云可视化工具 — 纯 numpy 实现，不依赖 open3d
输出 PLY 文件供 CloudCompare / MeshLab 查看
"""
import numpy as np
from pathlib import Path


# 类别颜色 (RGB 0-255)
CLASS_COLORS = {
    0: (180, 180, 180),   # table — 灰色
    1: (220, 50, 50),     # apple — 红色
    2: (255, 220, 50),    # banana — 黄色
}


def colorize_by_label(labels, class_colors=None):
    """
    标签 -> 颜色
    labels: (N,) int
    returns: (N, 3) uint8
    """
    if class_colors is None:
        class_colors = CLASS_COLORS
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for cls_id, rgb in class_colors.items():
        mask = labels == cls_id
        colors[mask] = rgb
    return colors


def save_ply_colored(filepath, points, labels, colors=None, normals=None):
    """
    保存带颜色的点云 PLY (ASCII)
    points: (N, 3)
    labels: (N,)  — 用于着色
    """
    if colors is None:
        colors = colorize_by_label(labels)

    n = len(points)
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int class\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            c = int(labels[i])
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {c}\n")


def compare_pred_gt(points, pred_labels, gt_labels, save_dir="vis"):
    """
    保存预测 vs 真实对比的 PLY 文件
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_ply_colored(save_dir / "pred.ply", points, pred_labels)
    save_ply_colored(save_dir / "gt.ply", points, gt_labels)

    # 误分类标红
    wrong = pred_labels != gt_labels
    vis_labels = gt_labels.copy()
    vis_colors = colorize_by_label(gt_labels)
    vis_colors[wrong] = [255, 0, 0]  # 红色标错
    save_ply_colored(save_dir / "error.ply", points, vis_labels, colors=vis_colors)

    acc = (~wrong).mean()
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved to: {save_dir}/pred.ply, gt.ply, error.ply")
