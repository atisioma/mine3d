"""
点云可视化脚本
用法:
  python scripts/vis.py                          # 随机可视化一个 raw + 一个 labeled
  python scripts/vis.py --raw data/raw/scene_00001.ply
  python scripts/vis.py --labeled data/train/scene_00001.ply
  python scripts/vis.py --compare data/raw/scene_00001.ply data/train/scene_00001.ply
  python scripts/vis.py --split train --count 5  # 随机展示 train 集 5 个
"""
import argparse
import numpy as np
from pathlib import Path

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


def read_ply_raw(filepath):
    """读取原始点云 (无标签): 返回 points, normals, colors"""
    n_points, header_end = 0, 0
    with open(filepath) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("element vertex"):
                n_points = int(s.split()[-1])
            if s == "end_header":
                header_end = i + 1
                break
    data_lines = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= header_end:
                data_lines.append(line)
                if len(data_lines) >= n_points:
                    break
    data = np.array([list(map(float, l.split())) for l in data_lines])
    return data[:, :3], data[:, 3:6], data[:, 6:9].astype(np.uint8)


def read_ply_labeled(filepath):
    """读取带标注点云: 返回 points, normals, colors, labels"""
    n_points, header_end = 0, 0
    with open(filepath) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("element vertex"):
                n_points = int(s.split()[-1])
            if s == "end_header":
                header_end = i + 1
                break
    data_lines = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= header_end:
                data_lines.append(line)
                if len(data_lines) >= n_points:
                    break
    data = np.array([list(map(float, l.split())) for l in data_lines])
    return data[:, :3], data[:, 3:6], data[:, 6:9].astype(np.uint8), data[:, 9].astype(np.int32)


CLASS_COLORS = {
    0: (0.71, 0.71, 0.71),   # table  灰
    1: (0.86, 0.12, 0.12),   # apple  红
    2: (1.00, 0.86, 0.20),   # banana 黄
}
CLASS_NAMES = {0: "table", 1: "apple", 2: "banana"}


def labels_to_colors(labels):
    colors = np.zeros((len(labels), 3))
    for cls_id, rgb in CLASS_COLORS.items():
        colors[labels == cls_id] = rgb
    return colors


def make_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def vis_open3d(pcd_list, title=""):
    assert HAS_O3D, "open3d 未安装"
    print(f"[{title}] 点云数: {len(pcd_list)}, 总点数: {sum(len(p.points) for p in pcd_list)}")
    if len(pcd_list) == 1:
        o3d.visualization.draw_geometries([pcd_list[0]], window_name=title)
    else:
        o3d.visualization.draw_geometries(pcd_list, window_name=title)


def save_debug_ply(filepath, points, colors):
    """open3d 不可用时, 保存为可直接打开的 PLY"""
    n = len(points)
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            r, g, b = (colors[i] * 255).astype(int)
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} {r} {g} {b}\n")
    print(f"  已保存: {filepath} (用 CloudCompare/MeshLab 打开)")


def main():
    parser = argparse.ArgumentParser(description="点云可视化")
    parser.add_argument("--raw", type=str, help="原始点云路径")
    parser.add_argument("--labeled", type=str, help="带标注点云路径")
    parser.add_argument("--compare", nargs=2, metavar=("RAW", "LABELED"), help="对比原始 vs 标注")
    parser.add_argument("--split", choices=["train", "val"], help="随机展示某个 split 的样本")
    parser.add_argument("--count", type=int, default=1, help="随机展示数量 (配合 --split)")
    parser.add_argument("--save", action="store_true", help="保存为 PLY 而非弹窗显示")
    args = parser.parse_args()

    pcds = []

    if args.compare:
        raw_p, raw_n, raw_c = read_ply_raw(args.compare[0])
        lab_p, lab_n, lab_c, lab_l = read_ply_labeled(args.compare[1])
        raw_colors = raw_c.astype(float) / 255.0
        lab_colors = labels_to_colors(lab_l)
        if args.save or not HAS_O3D:
            save_debug_ply("vis_raw.ply", raw_p, raw_colors)
            save_debug_ply("vis_labeled.ply", lab_p, lab_colors)
        else:
            pcds.append(make_pcd(raw_p, raw_colors))
            pcds.append(make_pcd(lab_p, lab_colors))
            vis_open3d(pcds, title="raw vs labeled")

    elif args.raw:
        pts, nrm, colors = read_ply_raw(args.raw)
        c = colors.astype(float) / 255.0
        if args.save or not HAS_O3D:
            save_debug_ply("vis_raw.ply", pts, c)
        else:
            pcds.append(make_pcd(pts, c))
            vis_open3d(pcds, title=f"raw: {Path(args.raw).name}")

    elif args.labeled:
        pts, nrm, colors, labels = read_ply_labeled(args.labeled)
        c = labels_to_colors(labels)
        if args.save or not HAS_O3D:
            save_debug_ply("vis_labeled.ply", pts, c)
        else:
            pcds.append(make_pcd(pts, c))
            vis_open3d(pcds, title=f"labeled: {Path(args.labeled).name}")

    elif args.split:
        if not HAS_O3D:
            print("错误: open3d 未安装, 请运行: uv add open3d")
            print("或使用 --save 模式保存 PLY 文件")
            return
        split_dir = Path(f"data/{args.split}")
        files = sorted(split_dir.glob("*.ply"))
        samples = np.random.choice(files, size=min(args.count, len(files)), replace=False)
        for f in samples:
            pts, nrm, colors, labels = read_ply_labeled(str(f))
            c = labels_to_colors(labels)
            dist = {0: 0, 1: 0, 2: 0}
            for l in labels:
                dist[int(l)] += 1
            dist_str = " ".join(f"{CLASS_NAMES[k]}={v}" for k, v in dist.items())
            if args.save:
                save_debug_ply(f"vis_{f.stem}.ply", pts, c)
            else:
                pcds.append(make_pcd(pts, c))
        if pcds:
            vis_open3d(pcds, title=f"{args.split} ({len(samples)} samples)")

    else:
        # 默认: 随机展示一个 raw + 一个 labeled
        raw_files = sorted(Path("data/raw").glob("*.ply"))
        train_files = sorted(Path("data/train").glob("*.ply"))
        rf = raw_files[np.random.randint(len(raw_files))]
        lf = train_files[np.random.randint(len(train_files))]

        r_p, r_n, r_c = read_ply_raw(str(rf))
        l_p, l_n, l_c, l_l = read_ply_labeled(str(lf))

        r_colors = r_c.astype(float) / 255.0
        l_colors = labels_to_colors(l_l)

        if args.save or not HAS_O3D:
            save_debug_ply("vis_raw.ply", r_p, r_colors)
            save_debug_ply("vis_labeled.ply", l_p, l_colors)
        else:
            pcds.append(make_pcd(r_p, r_colors))
            pcds.append(make_pcd(l_p, l_colors))
            vis_open3d(pcds, title="random raw + labeled")


if __name__ == "__main__":
    main()
