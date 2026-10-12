# -*- coding: utf-8 -*-
"""
阶段 2：RGB -> 检测分割 -> 深度 -> 掩膜×深度反投影 -> 三维点云

没有实机 RGB-D 相机时, 用 Depth-Anything-V2(度量深度版)从彩色图估计深度,
代替 Femto Mega 相机的深度图。反投影数学与真机完全一致:
    Z = depth(u,v)                    该像素的深度(米)
    X = (u - cx) * Z / fx             像素坐标 -> 相机坐标
    Y = (v - cy) * Z / fy

用法:
    python 02_pointcloud.py [图片路径] [检测文本]
输出:
    output/02_full_cloud.ply      全场景彩色点云(可用 MeshLab / CloudCompare 打开)
    output/02_target_cloud.ply    最大检出目标的点云(抓取分支的输入形态)
    output/02_preview.png         点云预览图
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD_CKPT = os.path.join(ROOT, "weights", "groundingdino_swint_ogc.pth")
SAM_CKPT = os.path.join(ROOT, "weights", "sam_vit_h_4b8939.pth")
DA_DIR = os.path.join(ROOT, "weights", "depth-anything-v2-metric-indoor-small")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 项目内的 GroundingDINO 配置副本(text_encoder_type 指向本地 bert)
GD_CFG = os.path.join(ROOT, "weights", "GroundingDINO_SwinT_OGC.py")

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?",
                    default=os.path.join(ROOT, "data", "beignets-task-guide.png"))
parser.add_argument("caption", nargs="?", default="cup . plate")
parser.add_argument("--box-thresh", type=float, default=0.30)
parser.add_argument("--text-thresh", type=float, default=0.25)
parser.add_argument("--fx-scale", type=float, default=1.2,
                    help="演示用内参: fx = fy = fx_scale * 图像宽度")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/4] 设备: {device}")

# ---------- 第一步: GroundingDINO 检测 + SAM 分割 (与阶段 1 相同) ----------
print("[2/4] 加载 GroundingDINO + SAM ...")
gd = load_model(GD_CFG, GD_CKPT, device=device)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(device)
sam_predictor = SamPredictor(sam)

image_source, image_tensor = load_image(args.image)
with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
    boxes, logits, phrases = predict(
        gd, image_tensor, args.caption,
        args.box_thresh, args.text_thresh, device=device,
    )

h, w = image_source.shape[:2]
print(f"      检出 {len(boxes)} 个目标: {phrases}")
if len(boxes) == 0:
    print("没有检出目标, 换个文本再试")
    sys.exit(0)

boxes_xyxy = boxes * torch.tensor([w, h, w, h], device=boxes.device)
boxes_xyxy = boxes_xyxy.cpu().numpy()
x_c, y_c, bw, bh = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
boxes_xyxy = np.stack([x_c - bw / 2, y_c - bh / 2, x_c + bw / 2, y_c + bh / 2], axis=1)

sam_predictor.set_image(image_source)
masks = []
for xyxy in boxes_xyxy:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    m, _, _ = sam_predictor.predict(point_coords=None, point_labels=None,
                                    box=box_np, multimask_output=False)
    masks.append(m[0])
masks = np.array(masks)  # (N, H, W) bool

# 释放检测分割模型显存, 给深度模型腾地方
del gd, sam, sam_predictor
torch.cuda.empty_cache()

# ---------- 第二步: 估计深度 (代替相机的 depth 图) ----------
print("[3/4] 加载 Depth-Anything-V2 (度量深度) ...")
from transformers import pipeline as hf_pipeline
pipe = hf_pipeline("depth-estimation", model=DA_DIR, device=device)
from PIL import Image
pil_img = Image.fromarray(image_source)
with torch.no_grad():
    depth_pred = pipe(pil_img)["predicted_depth"]      # (H, W) 张量, 单位: 米
depth = depth_pred.float().cpu().numpy()
if depth.shape != (h, w):  # 有的输出分辨率与原图不同, 对齐回来
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
print(f"      深度范围: {depth.min():.2f} ~ {depth.max():.2f} 米")
del pipe
torch.cuda.empty_cache()

# ---------- 第三步: 掩膜 × 有效深度 反投影成点云 ----------
print("[4/4] 反投影生成点云 ...")
fx = fy = args.fx_scale * w          # 演示用近似内参; 真机时换成 Femto Mega 标定值
cx, cy = w / 2.0, h / 2.0

u, v = np.meshgrid(np.arange(w), np.arange(h))
valid_scene = depth > 1e-6           # 真机深度图有空洞(0值), 统一用"有效深度"过滤
Z = depth[valid_scene]
X = (u[valid_scene] - cx) * Z / fx
Y = (v[valid_scene] - cy) * Z / fy
pts_scene = np.stack([X, Y, Z], axis=1)
colors_scene = image_source[valid_scene] / 255.0

# 最大检出目标(按框面积) -> 单独导出, 形态等同抓取分支的输入
areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
k = int(np.argmax(areas))
mask_k = masks[k] & valid_scene
Zk = depth[mask_k]
Xk = (u[mask_k] - cx) * Zk / fx
Yk = (v[mask_k] - cy) * Zk / fy
pts_k = np.stack([Xk, Yk, Zk], axis=1)
colors_k = image_source[mask_k] / 255.0
print(f"      目标 #{k+1} '{phrases[k]}' -> {len(pts_k)} 个点")

import open3d as o3d
def save_ply(path, pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)
    return path

p_full = save_ply(os.path.join(OUT_DIR, "02_full_cloud.ply"), pts_scene, colors_scene)
p_tgt = save_ply(os.path.join(OUT_DIR, "02_target_cloud.ply"), pts_k, colors_k)

# ---------- 预览图 (matplotlib, 无需 GUI) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(11, 5))
for j, (pts, colors, title) in enumerate([
        (pts_scene, colors_scene, "Full scene cloud"),
        (pts_k, colors_k, f"Target cloud: {phrases[k]}")]):
    ax = fig.add_subplot(1, 2, j + 1, projection="3d")
    stride = max(1, int(np.sqrt(len(pts) / 40000)))   # 降采样到约4万点便于绘图
    ax.scatter(pts[::stride, 0], pts[::stride, 1], pts[::stride, 2],
               c=colors[::stride], s=0.5, marker=".")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=-75, azim=-90)                  # 俯视视角, 与原图朝向一致
plt.tight_layout()
p_png = os.path.join(OUT_DIR, "02_preview.png")
plt.savefig(p_png, dpi=130)

print("完成!")
print(f"      全场景点云: {len(pts_scene)} 点 -> {p_full}")
print(f"      目标点云:   {len(pts_k)} 点 -> {p_tgt}")
print(f"      预览图:     {p_png}")
