"""
推理脚本 — 加载 PLY -> 模型推理 -> 保存预测结果
用法: python -m src.infer --ply data/val/scene_00001.ply
"""
import argparse
import torch
import numpy as np
from pathlib import Path

from src.dataset import read_ply
from src.model import build_model
from src.vis import save_ply_colored, CLASS_COLORS


def predict_ply(ply_path, checkpoint, device="cpu"):
    """
    对单个 PLY 文件做语义分割推理
    返回: points, pred_labels, pred_colors
    """
    # 读取
    points, normals, colors, gt_labels = read_ply(ply_path)

    # 归一化 (与训练一致)
    pts = points - points.mean(axis=0)
    max_val = np.abs(pts).max()
    if max_val > 0:
        pts = pts / max_val

    colors_f = colors.astype(np.float32) / 255.0

    # 构造输入
    feat = np.concatenate([pts, colors_f], axis=1)  # (N, 6)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)  # (1, N, 6)

    # 加载模型
    model = build_model(num_classes=3, in_ch=6).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(x)  # (1, N, 3)

    pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (N,)

    return points, pred, gt_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="输入 PLY 文件路径")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--out-dir", default="vis_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points, pred, gt = predict_ply(args.ply, args.checkpoint, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_ply_colored(out_dir / "pred.ply", points, pred)
    save_ply_colored(out_dir / "gt.ply", points, gt)

    acc = (pred == gt).mean()
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved: {out_dir}/pred.ply, gt.ply")


if __name__ == "__main__":
    main()
