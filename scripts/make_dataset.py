"""
v4 — 直接存相机坐标, 不反变换
  - 生成几何在世界坐标
  - 光照在世界坐标
  - 转相机坐标 → 可见性裁剪 → 深度噪声 → 直接存储
  - 法线也转相机坐标
"""
import numpy as np
import json
from pathlib import Path

NUM_CLASSES = 3
N_SAMPLES = 1000
N_TRAIN = 800
N_VAL = 200

def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-8)

def compute_normals_fast(points, k=12):
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    neighbors = points[idx]
    mean = neighbors.mean(axis=1, keepdims=True)
    diff = neighbors - mean
    cov = np.matmul(diff.transpose(0, 2, 1), diff) / k
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    normals[normals[:, 2] > 0] *= -1
    return normals.astype(np.float32)

def voxel_downsample(points, voxel_size=0.001):
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return idx

def apply_lighting(points, normals, base_colors, cam_pos):
    colors = base_colors.astype(np.float64)
    light_dirs = np.array([[0.5, 0.3, 1.0], [-0.4, -0.2, 0.8], [0.0, 0.0, 1.0]])
    light_ints = np.array([0.5, 0.2, 0.3])
    ldn = normalize(light_dirs)
    for i in range(3):
        diff = np.maximum(normals @ ldn[i], 0)
        colors += light_ints[i] * diff[:, None] * base_colors.astype(np.float64)
    view_dir = normalize(cam_pos[None, :] - points)
    for i in range(3):
        half_v = normalize(ldn[i][None, :] + view_dir)
        spec = np.power(np.maximum(np.einsum('ij,ij->i', normals, half_v), 0), 32)
        colors += light_ints[i] * 0.3 * spec[:, None] * 255
    colors = np.clip(0.25 * colors + colors, 0, 255)
    return colors.astype(np.uint8)

class VirtualCamera:
    def __init__(self, position, look_at, fov_deg=60, width=640, height=480):
        self.position = np.array(position, dtype=np.float64)
        self.fov = np.radians(fov_deg)
        self.width = width
        self.height = height
        forward = normalize(np.array(look_at, dtype=np.float64) - self.position).flatten()
        right = normalize(np.cross(forward, np.array([0, 0, 1.0]))).flatten()
        up = np.cross(right, forward)
        self.forward, self.right, self.up = forward, right, up
        fx = width / (2 * np.tan(self.fov / 2))
        self.intrinsic = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])

    def world_to_camera(self, points):
        rel = points - self.position
        return np.column_stack([rel @ self.right, rel @ self.up, rel @ self.forward])

    def normals_world_to_camera(self, normals):
        R = np.column_stack([self.right, self.up, self.forward])
        return (R @ normals.T).T

    def get_visible_mask(self, cam_points):
        z = cam_points[:, 2]
        valid = z > 0.05
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        u = np.full(len(z), -1.0)
        v = np.full(len(z), -1.0)
        u[valid] = fx * cam_points[valid, 0] / z[valid] + cx
        v[valid] = fy * cam_points[valid, 1] / z[valid] + cy
        return valid & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)

def make_table():
    top_w = np.random.uniform(0.5, 0.8)
    top_d = np.random.uniform(0.4, 0.6)
    top_h = 0.03
    leg_h = np.random.uniform(0.35, 0.45)
    leg_r = 0.022
    pts = []
    n_top = 4000
    x = np.random.uniform(-top_w/2, top_w/2, n_top)
    y = np.random.uniform(-top_d/2, top_d/2, n_top)
    grain = 0.005 * np.sin(x * 120) + 0.003 * np.sin(y * 80 + x * 20)
    z = top_h + grain + np.random.normal(0, 0.0003, n_top)
    pts.append(np.column_stack([x, y, z]))
    for side in range(4):
        ne = 300
        if side == 0: px, py = np.full(ne, top_w/2), np.random.uniform(-top_d/2, top_d/2, ne)
        elif side == 1: px, py = np.full(ne, -top_w/2), np.random.uniform(-top_d/2, top_d/2, ne)
        elif side == 2: px, py = np.random.uniform(-top_w/2, top_w/2, ne), np.full(ne, top_d/2)
        else: px, py = np.random.uniform(-top_w/2, top_w/2, ne), np.full(ne, -top_d/2)
        pts.append(np.column_stack([px, py, np.random.uniform(0, top_h, ne)]))
    n_bot = 800
    pts.append(np.column_stack([np.random.uniform(-top_w/2, top_w/2, n_bot),
                                 np.random.uniform(-top_d/2, top_d/2, n_bot),
                                 np.random.normal(0, 0.0003, n_bot)]))
    corners = [(top_w/2-0.04, top_d/2-0.04), (-top_w/2+0.04, top_d/2-0.04),
               (top_w/2-0.04, -top_d/2+0.04), (-top_w/2+0.04, -top_d/2+0.04)]
    for lx, ly in corners:
        nl = 600
        th = np.random.uniform(0, 2*np.pi, nl)
        r = leg_r + np.random.normal(0, 0.0008, nl)
        pts.append(np.column_stack([lx + r*np.cos(th), ly + r*np.sin(th),
                                     np.random.uniform(-leg_h, 0, nl)]))
    n_floor = 600
    pts.append(np.column_stack([
        np.random.uniform(-top_w/2-0.1, top_w/2+0.1, n_floor),
        np.random.uniform(-top_d/2-0.1, top_d/2+0.1, n_floor),
        np.full(n_floor, -leg_h) + np.random.normal(0, 0.002, n_floor)]))
    points = np.concatenate(pts).astype(np.float32)
    normals = compute_normals_fast(points)
    base = np.random.uniform(0.35, 0.55)
    gv = 0.12 * np.sin(points[:, 0] * 100 + points[:, 1] * 15)
    b = base + gv
    colors = np.column_stack([b * 180, b * 110, b * 60])
    colors += np.random.normal(0, 6, colors.shape)
    return points, normals, np.clip(colors, 30, 210).astype(np.uint8)

def make_apple():
    n = 4000
    radius = np.random.uniform(0.04, 0.055)
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0.05, np.pi-0.05, n)
    r = radius * (1 + 0.1*np.sin(2*theta) + 0.05*np.sin(3*phi)*np.sin(theta))
    r -= 0.15 * radius * np.exp(-4*(theta-0.1)**2)
    r -= 0.1 * radius * np.exp(-5*(theta-np.pi+0.15)**2)
    pts_body = np.column_stack([r*np.sin(theta)*np.cos(phi),
                                 r*np.sin(theta)*np.sin(phi),
                                 r*np.cos(theta)]) + np.random.normal(0, 0.0003, (n, 3))
    n_stem = 120
    t = np.linspace(0, 1, n_stem)
    th_s = np.random.uniform(0, 2*np.pi, n_stem)
    r_s = 0.002 * (1 - t*0.4)
    pts_stem = np.column_stack([r_s*np.cos(th_s), r_s*np.sin(th_s),
                                 radius*0.88 + t*np.random.uniform(0.02, 0.035)])
    points = np.vstack([pts_body, pts_stem])
    labels = np.concatenate([np.full(n, 1, np.int32), np.full(n_stem, 1, np.int32)])
    normals = compute_normals_fast(points)
    ti = theta / np.pi
    hs = np.random.uniform(-10, 10)
    r_c = 190 + 30*np.random.random(n) + hs
    g_c = np.where(ti < 0.25, 50 + 80*(0.25-ti)/0.25 + np.random.normal(0, 12, n),
                   np.where(ti > 0.75, 80 + 100*(ti-0.75)/0.25 + np.random.normal(0, 15, n),
                            25 + np.random.normal(0, 15, n)))
    b_c = np.where(ti < 0.25, 25 + np.random.normal(0, 8, n),
                   np.where(ti > 0.75, 20 + np.random.normal(0, 8, n), 12 + np.random.normal(0, 8, n)))
    hl = np.maximum(0, 1.0 - np.abs(ti - 0.3)*4) * 40
    body_c = np.column_stack([r_c + hl, g_c, b_c])
    stem_c = np.full((n_stem, 3), [70, 45, 18], dtype=np.float64)
    all_c = np.vstack([body_c, stem_c])
    n_spots = np.random.randint(8, 30)
    all_c[np.random.choice(n, n_spots, replace=False)] = [50, 25, 12]
    all_c += np.random.normal(0, 5, all_c.shape)
    return points, normals, np.clip(all_c, 5, 255).astype(np.uint8), labels

def make_banana():
    n = 4000
    length = np.random.uniform(0.14, 0.20)
    curve = np.random.uniform(0.05, 0.07)
    t = np.linspace(0, 1, n)
    cx = length * (t - 0.5)
    cy = curve * (1 - np.cos(t * np.pi * 0.75))
    cz = curve * 0.4 * np.sin(t * np.pi) + np.random.normal(0, 0.0008, n)
    profile = np.sin(t * np.pi) ** 0.55 * np.random.uniform(0.9, 1.1, n)
    ea = np.random.uniform(1.05, 1.25)
    eb = 1.0 / ea
    theta = np.random.uniform(0, 2*np.pi, n)
    rx = 0.014 * profile * ea * np.cos(theta)
    ry = 0.014 * profile * eb * np.sin(theta)
    points = np.column_stack([cx + rx, cy + ry, cz + ry * 0.2 * np.sin(t * np.pi)]) + np.random.normal(0, 0.0003, (n, 3))
    normals = compute_normals_fast(points)
    colors = np.column_stack([np.full(n, 235.0), np.full(n, 215.0), np.full(n, 35.0)])
    mask_green = t < 0.08
    mask_brown = t > 0.92
    bl_g = t[mask_green] / 0.08
    colors[mask_green] = np.column_stack([140*bl_g+70*(1-bl_g), 190*bl_g+130*(1-bl_g), 35*bl_g+25*(1-bl_g)])
    bl_b = (t[mask_brown] - 0.92) / 0.08
    colors[mask_brown] = np.column_stack([210-50*bl_b, 190-90*bl_b, 45-25*bl_b])
    n_spots = np.random.randint(0, 20)
    colors[np.random.choice(n, n_spots, replace=False)] = [130, 75, 22]
    colors += np.random.normal(0, 6, colors.shape)
    return points, normals, np.clip(colors, 8, 255).astype(np.uint8)

def place_objects(n_apples, n_bananas, table_top_w, table_top_d):
    placed = []
    positions = []
    for _ in range(n_apples):
        for _ in range(200):
            cx = np.random.uniform(-table_top_w/2*0.6, table_top_w/2*0.6)
            cy = np.random.uniform(-table_top_d/2*0.6, table_top_d/2*0.6)
            cz = np.random.uniform(0.05, 0.07)
            c = np.array([cx, cy, cz])
            if not placed or np.all(np.linalg.norm(np.array(placed)[:, :2] - c[:2], axis=1) >= 0.18):
                placed.append(c)
                positions.append(('apple', c))
                break
    for _ in range(n_bananas):
        for _ in range(200):
            cx = np.random.uniform(-table_top_w/2*0.5, table_top_w/2*0.5)
            cy = np.random.uniform(-table_top_d/2*0.5, table_top_d/2*0.5)
            cz = np.random.uniform(0.03, 0.05)
            c = np.array([cx, cy, cz])
            if not placed or np.all(np.linalg.norm(np.array(placed)[:, :2] - c[:2], axis=1) >= 0.22):
                placed.append(c)
                positions.append(('banana', c))
                break
    return positions

def make_scene(idx):
    np.random.seed(idx)
    cam_dist = np.random.uniform(0.5, 0.8)
    cam_height = np.random.uniform(0.25, 0.45)
    cam_angle = np.random.uniform(-0.5, 0.5)
    cam_pos = [cam_dist*np.cos(cam_angle), cam_dist*np.sin(cam_angle), cam_height]
    look_z = np.random.uniform(0.0, 0.15)
    cam = VirtualCamera(cam_pos, [0, 0, look_z], fov_deg=np.random.uniform(60, 80))

    top_w = np.random.uniform(0.5, 0.8)
    top_d = np.random.uniform(0.4, 0.6)

    all_pts, all_nrm, all_lbl, all_clr = [], [], [], []
    pts, nrm, clr = make_table()
    all_pts.append(pts); all_nrm.append(nrm); all_clr.append(clr)
    all_lbl.append(np.full(len(pts), 0, np.int32))

    n_apples = np.random.randint(1, 4)
    n_bananas = np.random.randint(1, 3)
    obj_positions = place_objects(n_apples, n_bananas, top_w, top_d)

    for obj_type, c in obj_positions:
        if obj_type == 'apple':
            pts, nrm, clr, lbl = make_apple()
        else:
            pts, nrm, clr = make_banana()
            lbl = np.full(len(pts), 2, np.int32)
        all_pts.append(pts + c); all_nrm.append(nrm); all_clr.append(clr); all_lbl.append(lbl)

    points = np.concatenate(all_pts).astype(np.float32)
    normals_w = np.concatenate(all_nrm).astype(np.float32)
    colors = np.concatenate(all_clr).astype(np.float64)
    labels = np.concatenate(all_lbl).astype(np.int32)

    colors = apply_lighting(points, normals_w, colors, cam.position)

    cam_pts = cam.world_to_camera(points)
    cam_nrm = cam.normals_world_to_camera(normals_w)
    vis = cam.get_visible_mask(cam_pts)

    cam_pts = cam_pts[vis]
    cam_nrm = cam_nrm[vis]
    colors = colors[vis]
    labels = labels[vis]

    cam_pts[:, 2] += np.random.normal(0, 0.001, len(cam_pts))

    flip = cam_nrm[:, 2] > 0
    cam_nrm[flip] *= -1

    idx_d = voxel_downsample(cam_pts, 0.001)
    return cam_pts[idx_d].astype(np.float32), cam_nrm[idx_d].astype(np.float32), \
           colors[idx_d].astype(np.uint8), labels[idx_d].astype(np.int32)

def write_ply(filepath, points, normals, colors, labels=None):
    n = len(points)
    lines = [f"ply\nformat ascii 1.0\nelement vertex {n}\n"]
    for p in ["float x", "float y", "float z", "float nx", "float ny", "float nz",
              "uchar red", "uchar green", "uchar blue"]:
        lines.append(f"property {p}\n")
    if labels is not None:
        lines.append("property uchar class\n")
    lines.append("end_header\n")
    with open(filepath, "w") as f:
        f.write("".join(lines))
        data = np.column_stack([points, normals, colors]) if labels is None else np.column_stack([points, normals, colors, labels])
        for row in data:
            f.write(" ".join(f"{v:.6f}" if i < 6 else str(int(v)) for i, v in enumerate(row)) + "\n")

def main():
    raw_dir, train_dir, val_dir = Path("data/raw"), Path("data/train"), Path("data/val")
    for d in [raw_dir, train_dir, val_dir]:
        d.mkdir(parents=True, exist_ok=True)
    indices = np.random.permutation(N_SAMPLES)
    train_ids = set(indices[:N_TRAIN].tolist())
    val_ids = set(indices[N_TRAIN:].tolist())
    for i in range(N_SAMPLES):
        points, normals, colors, labels = make_scene(i)
        name = f"scene_{i:05d}.ply"
        write_ply(str(raw_dir / name), points, normals, colors)
        if i in train_ids:
            write_ply(str(train_dir / name), points, normals, colors, labels)
        elif i in val_ids:
            write_ply(str(val_dir / name), points, normals, colors, labels)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{N_SAMPLES}] {name}  pts={len(points)}")
    meta = {"name": "IndoorSeg", "num_classes": NUM_CLASSES,
            "classes": {"0": "table", "1": "apple", "2": "banana"},
            "num_samples": N_SAMPLES, "format": "PLY ASCII",
            "camera": "simulated RGB-D, low angle, camera coords",
            "split": {"train": N_TRAIN, "val": N_VAL}}
    with open("data/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nv4 done! raw/{N_SAMPLES} train/{N_TRAIN} val/{N_VAL}")

if __name__ == "__main__":
    main()
