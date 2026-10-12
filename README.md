# 桌面物体抓取与按压系统 · 感知与决策算法独立实现（项目包）

从 RGB 图像到可执行动作位姿的完整软件管线，已在 6GB 消费级 GPU（RTX 3060 Laptop）上验证通过。

## 目录内容

```
├── scripts/                  四个阶段脚本（中文注释）
│   ├── 01_detect_segment.py     文本 → GroundingDINO 检测 → SAM 分割
│   ├── 02_pointcloud.py         掩膜 × 深度反投影 → 三维点云
│   ├── 03_press_point.py        RANSAC 平面 + PCA 法向 + 边界余量按压点
│   └── 04_grasp_candidates.py   对跖点抓取候选采样（GraspNet 格式兼容）
├── data/                     测试图片
├── output/                   全部实测结果（PLY 点云 / PNG 可视化 / JSON 位姿）
├── requirements.txt          精确依赖版本（pip freeze 导出）
├── 复现总结.md               管线原理、实测数据、踩坑记录、实验室断点接续清单
└── 简历写法.md               项目经历简历表述与面试应答要点
```

## 环境与运行

- Python 3.10（conda 环境），PyTorch 2.5.1 + cu121，详见 requirements.txt
- 国内安装源：清华 PyPI（HTTPS）+ 阿里云 pytorch-wheels（仅 HTTP）
- 运行顺序：

```powershell
python scripts\01_detect_segment.py data\beignets-task-guide.png "cup . plate"
python scripts\02_pointcloud.py data\beignets-task-guide.png "cup . plate"
python scripts\03_press_point.py
python scripts\04_grasp_candidates.py
```

## 权重文件（未包含在本包，共约 3.6GB）

| 文件 | 大小 | 下载源 |
|---|---|---|
| groundingdino_swint_ogc.pth | 662MB | hf-mirror.com / IDEA 官方 |
| sam_vit_h_4b8939.pth | 2.4GB | hf-mirror.com（镜像 djghosh/wesonmith sam_vit_h） |
| bert-base-uncased/（5 个文件） | 420MB | hf-mirror.com |
| depth-anything-v2-metric-indoor-small/ | 95MB | hf-mirror.com |

下载后放入 `weights/` 目录，目录结构与 `复现总结.md` 第 2 节一致。
`weights/GroundingDINO_SwinT_OGC.py` 配置副本需包含（指明本地 bert 路径），
运行前设置环境变量 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 即可全离线运行。

## 关键实测指标

- 检测分割：512×512 图检出 11~12 目标，SAM 掩膜像素级贴合
- 点云：单帧 26.2 万点（全场景），深度范围 0.55~1.80 m
- 按压分支：RANSAC 内点率 51%，平面 RMS 3.2mm，边界余量 17mm
- 抓取分支：10.9 万近邻点对 → 7312 对跖候选 → NMS 后 Top-6（Top-1 得分 0.94）
