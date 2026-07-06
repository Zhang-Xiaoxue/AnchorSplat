# AnchorSplat

**中文** | [English](README.md)

**AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors**  
论文 `AnchorSplat_arkiv` 的官方实现。

[[Paper](https://arxiv.org/abs/2604.07053)]

<p align="center">
  <img src="readme/teaser.png" alt="AnchorSplat teaser" width="100%">
</p>

## 简介

AnchorSplat 是一个面向场景级重建的前馈式 3D Gaussian Splatting 框架。它不再为每个像素预测一个 Gaussian，而是利用几何先验生成稀疏 3D anchors，并预测 anchor-aligned Gaussians，从而减少冗余、提升多视角一致性。

本代码主要在 Ascend NPU 上运行和测试。模型基于 PyTorch 实现，可较方便迁移到 GPU；本文档仅保留 NPU 运行流程。

## 文件结构

当前仓库主要路径如下：

```text
AnchorSplat/
  README.md                 # 英文 README，GitHub 默认展示
  README_zh.md              # 中文 README
  LICENSE
  readme/                   # README 图片资源
    teaser.png
    pipeline.png
    psnr_time.png
  src/
    depth_anything_3/
      configs/
        anchorsplat_mix.yaml
      model/
      utils/
    train/
      trainer_mix.py
      dataset_mix.py
      loss.py
      data/                 # ScanNet++ / ARKitScenes 场景列表
```

## 方法流程

<p align="center">
  <img src="readme/pipeline.png" alt="AnchorSplat pipeline" width="100%">
</p>

- **Anchor 预测器**：从多视角图像中预测相机位姿、深度和 3D 几何先验。
- **Gaussian 解码器**：将多视角特征投影到 anchors 上，并预测 anchor-aligned 3D Gaussians。
- **Gaussian 精炼器**：利用渲染误差进一步优化 Gaussian 属性，提升重建质量。

<p align="center">
  <img src="readme/psnr_time.png" alt="PSNR and reconstruction time" width="72%">
</p>

## 环境

```bash
git clone https://github.com/Zhang-Xiaoxue/AnchorSplat.git
cd AnchorSplat

conda create -n anchorsplat python=3.10 -y
conda activate anchorsplat

# 根据 Ascend CANN 版本安装匹配的 PyTorch、torchvision 和 torch-npu。

pip install addict einops omegaconf opencv-python pillow imageio matplotlib tqdm typer \
  huggingface_hub torchmetrics plyfile trimesh moviepy gradio fastapi uvicorn \
  requests scipy scikit-learn open3d pycolmap evo easydict pillow-heif

export PYTHONPATH=$PWD/src:$PYTHONPATH
```

NPU 渲染依赖 Ascend 运行环境和 NPU Gaussian rasterization 算子，包括 `acl` 与 `meta_gauss_render`。

## 数据格式

### AnchorSplat-style data

用于 ScanNet++ 和 ARKitScenes：

```text
scene/
  images/
  viewInfo.npz
```

`viewInfo.npz` 应包含：

```text
image_filenames
T_w2c
K_norm
```

场景列表：

```text
src/train/data/scannetpp_train.txt
src/train/data/scannetpp_val.txt
src/train/data/arkit_train.txt
src/train/data/arkit_val.txt
```

### Reliev3R-style data

用于 DL3DV 和 RealEstate10K-style 数据：

```text
sub_xxx/
  scene/
    rgb/
    scene.npy
```

`scene.npy` 为 Python dict，应包含：

```text
intr_mat
extr_mat
```

每个场景建议不少于 `2 * num_views` 张图像；`num_views` 在 `src/depth_anything_3/configs/anchorsplat_mix.yaml` 中配置。

## 模型权重

训练配置默认读取：

```yaml
model:
  depth_net: ~/ckpts/DA3-GIANT-1.1
```

训练前请修改 `src/depth_anything_3/configs/anchorsplat_mix.yaml` 中的权重和数据路径，或在命令行中覆盖。

## Release Plan

- [ ] 训练权重上传后 release。

## 训练

```bash
torchrun --nproc_per_node=<num_npus> src/train/trainer_mix.py \
  --config src/depth_anything_3/configs/anchorsplat_mix.yaml \
  --output outputs/anchorsplat_mix
```

## 测试与推理

```bash
torchrun --nproc_per_node=<num_npus> src/train/trainer_mix.py \
  --config src/depth_anything_3/configs/anchorsplat_mix.yaml \
  --output outputs/anchorsplat_mix \
  --ckpt outputs/anchorsplat_mix/ckpt/epoch_xxxx.ckpt \
  --test
```

结果会保存到 `outputs/anchorsplat_mix/test`，包含指标、可视化结果，以及配置中开启后的 Gaussian/video 导出。

## 备注

代码中保留了用于兼容的 `gsplat` renderer fallback，但本仓库文档按 NPU-first 版本维护。

## 致谢

本实现基于 [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) codebase 修改完成，感谢 DA3 作者和社区贡献者开源相关代码。

## 引用

```bibtex
@article{zhang2026anchorsplat,
  title={AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors},
  author={Zhang, Xiaoxue and Zheng, Xiaoxu and Yin, Yixuan and Zhao, Tiao and Tang, Kaihua and Mi, Michael Bi and Xu, Zhan and Chen, Dave Zhenyu},
  journal={arXiv preprint arXiv:2604.07053},
  year={2026}
}
```
