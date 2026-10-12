# -*- coding: utf-8 -*-
"""
阶段 1 最小示例：一张图片 -> GroundingDINO 按文本检测物体 -> SAM 用检测框做精确分割
用法:
    python 01_detect_segment.py [图片路径] [检测文本]
示例:
    python 01_detect_segment.py data/test.jpg "paper bag . bottle . cup"
检测文本规则: 多个类别用英文句号 " . " 分隔, 类别用英文单词
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch

# 让 Windows 控制台能打印中文
sys.stdout.reconfigure(encoding="utf-8")

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GD_CKPT = os.path.join(ROOT, "weights", "groundingdino_swint_ogc.pth")
SAM_CKPT = os.path.join(ROOT, "weights", "sam_vit_h_4b8939.pth")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# GroundingDINO 配置: 项目内的副本, 其中 text_encoder_type 指向本地 bert 目录
GD_CFG = os.path.join(ROOT, "weights", "GroundingDINO_SwinT_OGC.py")

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", default=os.path.join(ROOT, "data", "test.jpg"))
parser.add_argument("caption", nargs="?",
                    default="paper bag . bottle . cup . box . book . bowl . object")
parser.add_argument("--box-thresh", type=float, default=0.30)   # 检测框置信度阈值
parser.add_argument("--text-thresh", type=float, default=0.25)  # 文本匹配阈值
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/4] 设备: {device}")

# ---------- 加载两个模型 ----------
print("[2/4] 加载 GroundingDINO (SwinT) ...")
gd = load_model(GD_CFG, GD_CKPT, device=device)

print("[3/4] 加载 SAM ViT-H ...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
sam.to(device)
sam_predictor = SamPredictor(sam)

# ---------- GroundingDINO 检测 ----------
print(f"[4/4] 检测: {args.image}")
print(f"      文本提示: {args.caption}")
image_source, image_tensor = load_image(args.image)  # numpy(RGB) 和 归一化张量

with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
    boxes, logits, phrases = predict(
        gd, image_tensor, args.caption,
        args.box_thresh, args.text_thresh, device=device,
    )

h, w = image_source.shape[:2]
print(f"      图像尺寸: {w} x {h}, 检出 {len(boxes)} 个目标: {phrases}")

if len(boxes) == 0:
    print("没有检出任何物体, 可以换更简单的文本(如 'object')或调低 --box-thresh")
    sys.exit(0)

# cxcywh 归一化 -> 原图上的 xyxy 像素坐标
boxes_xyxy = boxes * torch.tensor([w, h, w, h], device=boxes.device)
boxes_xyxy = boxes_xyxy.cpu().numpy()
x_c, y_c, bw, bh = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
boxes_xyxy = np.stack([x_c - bw / 2, y_c - bh / 2, x_c + bw / 2, y_c + bh / 2], axis=1)

# ---------- SAM: 用每个检测框做 box 提示分割 ----------
sam_predictor.set_image(image_source)  # 内部计算图像特征, 一张图只算一次

vis = image_source.copy()
rng = np.random.default_rng(42)
colors = rng.integers(60, 230, size=(len(boxes_xyxy), 3))

for i, (xyxy, phrase, score) in enumerate(zip(boxes_xyxy, phrases, logits)):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    # SAM 的 predict 接口收 numpy 数组(内部自己转张量), 不要传 torch 张量
    box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    masks, mask_scores, _ = sam_predictor.predict(
        point_coords=None, point_labels=None,
        box=box_np, multimask_output=False,
    )
    mask = masks[0]
    color = colors[i]
    # 半透明掩膜
    layer = vis.copy()
    layer[mask] = color
    vis = cv2.addWeighted(vis, 0.55, layer, 0.45, 0)
    # 掩膜轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color.tolist(), 2)
    # 检测框 + 标签
    cv2.rectangle(vis, (x1, y1), (x2, y2), color.tolist(), 2)
    label = f"{phrase} {float(score):.2f}"
    cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.tolist(), 2)
    print(f"      #{i+1} {label}  box=({x1},{y1},{x2},{y2})  "
          f"mask_pixels={int(mask.sum())}")

out_path = os.path.join(OUT_DIR, "01_result.png")
cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"完成! 结果已保存: {out_path}")
