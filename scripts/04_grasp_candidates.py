# -*- coding: utf-8 -*-
"""
阶段 4：目标点云 -> 对跖点(Antipodal)抓取候选 -> 机械臂可执行的抓取位姿

会议项目用 GraspNet-1Billion 网络生成抓取候选(权重在网盘, 家里下不了),
这里用经典几何方法"对跖点采样"达到同样目的:
    两根手指要夹稳物体, 需要找到一对表面点满足:
      ① 两个点的法向正好相对(一个朝左一个朝右) —— "对跖"
      ② 两点间距在夹爪开口范围内(本例 10~80 mm)
      ③ 接近方向(手指伸入的方向)取两点平均法向
    满足①②的点对就是一个几何上合理的抓取候选, 再按质量打分去重。

输出候选格式与 GraspNet 一致: 位置 + 闭合轴 + 接近方向 + 开口宽度 + 得分,
到实验室后可直接换成 GraspNet 网络的输出。

用法:
    python 04_grasp_candidates.py [目标点云.ply]
输出:
    output/04_grasp_candidates.json   前 K 个抓取候选
    output/04_grasp_preview.png       点云 + 抓取位姿可视化
"""
import os
import sys
import json

import numpy as np
import open3d as o3d

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 参数(夹爪几何, 按项目机械臂参数设) ----------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cloud", nargs="?",
                    default=os.path.join(OUT_DIR, "02_target_cloud.ply"))
parser.add_argument("--width-min", type=float, default=0.010)  # 最小开口 10mm
parser.add_argument("--width-max", type=float, default=0.080)  # 最大开口 80mm
parser.add_argument("--nms-dist", type=float, default=0.025)   # 去重距离 25mm
parser.add_argument("--topk", type=int, default=6)             # 保留前几个
args = parser.parse_args()

# ---------- 读取并清理点云, 估计法向 ----------
print("[1/4] 读取点云:", args.cloud)
pcd = o3d.io.read_point_cloud(args.cloud)
pcd = pcd.voxel_down_sample(0.003)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

print("[2/4] 估计法向 (相机在原点, 法向朝向相机) ...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.03, max_nn=50))
pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))
pts = np.asarray(pcd.points)
nrm = np.asarray(pcd.normals)
print(f"      点数: {len(pts)}")

# ---------- 对跖点对采样 ----------
print("[3/4] 对跖点对采样 ...")
# 用 KDTree 找出所有间距 < 最大开口 的点对
tree = o3d.geometry.KDTreeFlann(pcd)
pairs_i, pairs_j = [], []
for i in range(len(pts)):
    _, idx, _ = tree.search_knn_vector_3d(pts[i], 60)      # 每点取60近邻
    for j in idx:
        if j > i:
            pairs_i.append(i); pairs_j.append(j)
pairs_i = np.array(pairs_i); pairs_j = np.array(pairs_j)
print(f"      近邻点对: {len(pairs_i)}")

d_vec = pts[pairs_j] - pts[pairs_i]                        # i -> j 向量
width = np.linalg.norm(d_vec, axis=1)
ok = (width >= args.width_min) & (width <= args.width_max)
pairs_i, pairs_j, d_vec, width = pairs_i[ok], pairs_j[ok], d_vec[ok], width[ok]

axis = d_vec / width[:, None]                              # 闭合轴(单位向量)
n1, n2 = nrm[pairs_i], nrm[pairs_j]

# 对跖条件: i 的法向朝向 j, j 的法向朝向 i (两法向相对)
cond1 = np.einsum("ij,ij->i", n1, d_vec) > 0
cond2 = np.einsum("ij,ij->i", n2, -d_vec) > 0
anti = cond1 & cond2
pairs_i, pairs_j = pairs_i[anti], pairs_j[anti]
axis, width, n1, n2 = axis[anti], width[anti], n1[anti], n2[anti]
print(f"      满足对跖+开口条件的候选: {len(pairs_i)}")

if len(pairs_i) == 0:
    print("没有可用候选(点云太小或太薄), 换个目标试试")
    sys.exit(0)

# ---------- 打分 + 非极大值抑制(NMS) ----------
print("[4/4] 打分与去重 ...")
oppo = np.einsum("ij,ij->i", n1, -n2)                      # 法向相对程度 [-1,1], 越大越"对跖"
approach = -(n1 + n2)                                      # 接近方向 = 平均法向反向(从上往下夹)
approach /= np.linalg.norm(approach, axis=1)[:, None] + 1e-9
# 得分 = 对跖程度为主, 开口适中略加分
score = 0.8 * oppo + 0.2 * (1 - width / args.width_max)

order = np.argsort(-score)
keep = []
for idx in order:
    p = pts[pairs_i[idx]]
    a = axis[idx]
    dup = False
    for k in keep:                                         # 位置近 + 闭合轴近平行 => 视为重复
        if np.linalg.norm(pts[pairs_i[k]] - p) < args.nms_dist and \
           abs(np.dot(axis[k], a)) > 0.87:
            dup = True; break
    if not dup:
        keep.append(idx)
    if len(keep) >= args.topk:
        break

cands = []
for rank, k in enumerate(keep):
    pi, pj = pairs_i[k], pairs_j[k]
    cands.append({
        "rank": rank + 1,
        "score": round(float(score[k]), 3),
        "position_m": [float(v) for v in (pts[pi] + pts[pj]) / 2],  # 抓取中心
        "closing_axis": [float(v) for v in axis[k]],                # 手指闭合方向
        "approach_dir": [float(v) for v in approach[k]],            # 手臂进给方向
        "width_m": round(float(width[k]), 4),                       # 开口宽度
        "contact_i": [float(v) for v in pts[pi]],
        "contact_j": [float(v) for v in pts[pj]],
    })
    c = cands[-1]
    print(f"      #{rank+1} score={c['score']} width={c['width_m']*1000:.0f}mm "
          f"center=[{c['position_m'][0]:.3f},{c['position_m'][1]:.3f},{c['position_m'][2]:.3f}]")

with open(os.path.join(OUT_DIR, "04_grasp_candidates.json"), "w", encoding="utf-8") as f:
    json.dump({"gripper": {"width_min_m": args.width_min, "width_max_m": args.width_max},
               "candidates": cands}, f, ensure_ascii=False, indent=2)

# ---------- 可视化 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
sub = pts[::max(1, len(pts) // 15000)]
ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c="lightgray", s=0.6, depthshade=False)

cmap = plt.get_cmap("turbo")
for c in cands:
    col = cmap((c["rank"] - 1) / max(1, len(cands) - 1))
    ci = np.array(c["contact_i"]); cj = np.array(c["contact_j"])
    ctr = np.array(c["position_m"]); ap = np.array(c["approach_dir"])
    ax.scatter(*ci, color=col, s=25); ax.scatter(*cj, color=col, s=25)
    ax.plot([ci[0], cj[0]], [ci[1], cj[1]], [ci[2], cj[2]], color=col, linewidth=1.5)
    ax.quiver(*ctr, *(ap * 0.06), color=col, linewidth=1.5,
              arrow_length_ratio=0.25)
    ax.text(*ctr, f"#{c['rank']}", color=col, fontsize=8)

ax.set_title("Grasp candidates: line=closing axis, arrow=approach", fontsize=10)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.tight_layout()
p_png = os.path.join(OUT_DIR, "04_grasp_preview.png")
plt.savefig(p_png, dpi=130)

print("完成!")
print(f"      候选 JSON: {os.path.join(OUT_DIR, '04_grasp_candidates.json')}")
print(f"      预览图:    {p_png}")
