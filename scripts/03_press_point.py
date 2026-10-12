# -*- coding: utf-8 -*-
"""
阶段 3：目标点云 -> 几何分析 -> 选取按压点 (按压分支核心算法)

对应会议项目里"按压分支"的三步:
    1. RANSAC 平面拟合 —— 在目标点云里找出主平面(比如袋面/盘面),
       得到平面方程 ax+by+cz+d=0 和内点集合
    2. PCA 求法向与主轴 —— 内点做主成分分析, 得到表面法向(机械臂接近方向)
       和平面内两个主方向(夹爪朝向参考), 并输出平直度(RMS 距离)
    3. 边界余量最大的内部点 —— 把内点投影到平面栅格上, 用距离变换找出
       "离边界最远"的内部点作为按压点(避免按到边缘把物体推倒)

用法:
    python 03_press_point.py [目标点云.ply]
输出:
    output/03_press_preview.png   分析过程可视化(边界余量热图 + 3D结果)
    output/03_press_result.json   平面/法向/主轴/按压点坐标
"""
import os
import sys
import json

import cv2
import numpy as np
import open3d as o3d

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 参数 ----------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cloud", nargs="?",
                    default=os.path.join(OUT_DIR, "02_target_cloud.ply"))
parser.add_argument("--voxel", type=float, default=0.004,
                    help="体素降采样边长(米)")
parser.add_argument("--plane-dist", type=float, default=0.006,
                    help="RANSAC 平面内点距离阈值(米)")
parser.add_argument("--grid-cell", type=float, default=0.004,
                    help="边界余量栅格的格子边长(米)")
args = parser.parse_args()

# ---------- 读取并清理点云 ----------
print("[1/4] 读取点云:", args.cloud)
pcd = o3d.io.read_point_cloud(args.cloud)
pts = np.asarray(pcd.points)
print(f"      原始点数: {len(pts)}")

pcd = pcd.voxel_down_sample(args.voxel)                    # 体素降采样
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pts = np.asarray(pcd.points)
print(f"      降采样+去离群后: {len(pts)} 点")

# ---------- 第一步: RANSAC 平面拟合 ----------
print("[2/4] RANSAC 平面拟合 ...")
plane, inlier_idx = pcd.segment_plane(distance_threshold=args.plane_dist,
                                      ransac_n=3, num_iterations=1000)
a, b, c, d = np.array(plane, dtype=np.float64)             # ax+by+cz+d=0, (a,b,c)已归一
inliers = pts[inlier_idx]

# 距离统计 -> 平直度
residual = np.abs(inliers @ np.array([a, b, c]) + d)
print(f"      内点: {len(inliers)}/{len(pts)} ({100*len(inliers)/len(pts):.0f}%), "
      f"RMS={np.sqrt((residual**2).mean())*1000:.1f} mm")
print(f"      平面: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

# 法向统一朝向相机原点(0,0,0)方向 —— 真机上即"朝向相机"
n = np.array([a, b, c])
if n @ inliers.mean(axis=0) + d > 0:                       # 让 n 指向原点一侧
    n, d = -n, -d

# ---------- 第二步: PCA 主轴 ----------
print("[3/4] PCA 求主轴 ...")
center = inliers.mean(axis=0)
X = inliers - center
cov = X.T @ X / len(X)
w, vec = np.linalg.eigh(cov)                               # 特征值升序
ax1, ax2 = vec[:, -1], vec[:, -2]                          # 平面内两个主方向
print(f"      法向(接近方向): [{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]")
print(f"      主轴1(展向):   [{ax1[0]:.2f}, {ax1[1]:.2f}, {ax1[2]:.2f}] σ={np.sqrt(w[-1])*1000:.0f} mm")
print(f"      主轴2(走向):   [{ax2[0]:.2f}, {ax2[1]:.2f}, {ax2[2]:.2f}] σ={np.sqrt(w[-2])*1000:.0f} mm")

# ---------- 第三步: 边界余量 -> 按压点 ----------
print("[4/4] 边界余量分析 ...")
uv = np.stack([inliers @ ax1, inliers @ ax2], axis=1)      # 投影到平面2D坐标
uv_min, uv_max = uv.min(axis=0), uv.max(axis=0)
gw = int(np.ceil((uv_max[0] - uv_min[0]) / args.grid_cell)) + 1
gh = int(np.ceil((uv_max[1] - uv_min[1]) / args.grid_cell)) + 1
grid = np.zeros((gh, gw), np.uint8)
gi = np.clip(((uv - uv_min) / args.grid_cell).astype(int), 0, [gw - 1, gh - 1])
grid[gi[:, 1], gi[:, 0]] = 255

dist = cv2.distanceTransform((grid > 0).astype(np.uint8),
                             cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
fy_, fx_ = np.unravel_index(np.argmax(dist), dist.shape)   # 离边界最远的格子
margin_mm = dist[fy_, fx_] * args.grid_cell * 1000
cell_pts = inliers[(gi[:, 0] == fx_) & (gi[:, 1] == fy_)]
press = cell_pts.mean(axis=0) if len(cell_pts) else None
if press is None:  # 该格子无实测点时, 取离格子中心最近的点
    cc = uv_min + ([fx_ + 0.5, fy_ + 0.5]) * args.grid_cell
    press = inliers[np.argmin(np.linalg.norm(uv - cc, axis=1))]
print(f"      按压点: [{press[0]:.3f}, {press[1]:.3f}, {press[2]:.3f}] m, "
      f"边界余量 {margin_mm:.0f} mm")

# ---------- 保存结果 ----------
result = {
    "plane_abc_d": [float(v) for v in (a, b, c, d)],
    "normal_approach": [float(v) for v in n],
    "pca_axis1": [float(v) for v in ax1], "pca_axis2": [float(v) for v in ax2],
    "pca_sigma_mm": [float(np.sqrt(w[-1]) * 1000), float(np.sqrt(w[-2]) * 1000)],
    "plane_rms_mm": float(np.sqrt((residual**2).mean()) * 1000),
    "inlier_ratio": float(len(inliers) / len(pts)),
    "press_point_m": [float(v) for v in press],
    "press_margin_mm": float(margin_mm),
    "approach_dir": [float(v) for v in (-n)],              # 机械臂从 -n 方向压下
}
with open(os.path.join(OUT_DIR, "03_press_result.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# ---------- 可视化 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(11, 5))

# 左: 平面内 2D 边界余量热图
ax = fig.add_subplot(1, 2, 1)
ax.imshow(dist, origin="lower", cmap="viridis",
          extent=[uv_min[0], uv_max[0], uv_min[1], uv_max[1]])
ax.scatter(uv[:, 0], uv[:, 1], s=0.3, c="white", alpha=0.25)
ax.scatter(press @ ax1, press @ ax2, c="red", s=80, marker="x", linewidths=2)
ax.set_title(f"Boundary margin (press = red x, {margin_mm:.0f} mm)", fontsize=10)
ax.set_xlabel("plane axis 1 (m)"); ax.set_ylabel("plane axis 2 (m)")

# 右: 3D 结果(点云 + 法向箭头 + 按压点)
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
sub = pts[::max(1, len(pts) // 20000)]
ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], cmap="viridis", s=0.5)
L = 0.06
ax3.quiver(*press, *(n * L), color="red", linewidth=2, label="normal")
ax3.scatter(*press, c="red", s=60, marker="*", label="press point")
ax3.set_title("Plane / normal / press point", fontsize=10)
ax3.legend(loc="upper left", fontsize=8)
ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
plt.tight_layout()
p_png = os.path.join(OUT_DIR, "03_press_preview.png")
plt.savefig(p_png, dpi=130)

print("完成!")
print(f"      结果 JSON: {os.path.join(OUT_DIR, '03_press_result.json')}")
print(f"      预览图:    {p_png}")
