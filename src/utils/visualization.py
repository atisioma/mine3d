"""
点云可视化工具 — 源自 LitePT/utils/visualization.py
依赖: open3d (可选), numpy, torch
"""
import os
import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def get_point_cloud(coord, color=None, verbose=True):
    """创建 open3d 点云并可视化"""
    assert o3d is not None, "open3d not installed"
    if not isinstance(coord, list):
        coord = [coord]
        if color is not None:
            color = [color]

    pcd_list = []
    for i in range(len(coord)):
        coord_ = to_numpy(coord[i])
        color_ = to_numpy(color[i]) if color is not None else np.ones_like(coord_)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord_)
        pcd.colors = o3d.utility.Vector3dVector(color_)
        pcd_list.append(pcd)
    if verbose:
        o3d.visualization.draw_geometries(pcd_list)
    return pcd_list


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    """保存点云到 PLY 文件"""
    assert o3d is not None, "open3d not installed"
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    coord = to_numpy(coord)
    color = to_numpy(color) if color is not None else np.ones_like(coord)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None):
    """保存 3D bounding boxes 到 PLY"""
    assert o3d is not None, "open3d not installed"
    bboxes_corners = to_numpy(bboxes_corners)
    points = bboxes_corners.reshape(-1, 3)
    box_lines = np.array([
        [0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,0],[0,4],[1,5],[2,6],[3,7]
    ])
    lines = np.concatenate([box_lines + i * 8 for i in range(len(bboxes_corners))])
    color_arr = np.array([color for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color_arr)
    o3d.io.write_line_set(file_path, line_set)
    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def min_max_normalize(x):
    """归一化到 [0, 1]"""
    high = x.max(dim=0).values.float()
    low = x.min(dim=0).values.float()
    x_normalized = (x.float() - low) / (high - low + 1e-8)
    x_normalized[x_normalized.isnan() | x_normalized.isinf()] = 0
    return x_normalized
