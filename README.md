# PRDiT: Pixel-Level Residual Diffusion Transformer for Scalable 3D CT Volume Generation

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=bWtRZQ1rm2)
[![Poster](https://img.shields.io/badge/Poster-ICLR%202026-8A2BE2)](https://iclr.cc/media/PosterPDFs/ICLR%202026/10008602.png?t=1774447885.2973316)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **PRDiT** — *Pixel-Level Residual Diffusion Transformer* — a scalable approach for 3D CT volume generation, accepted at **ICLR 2026**.

## 📑 Table of Contents

- [Paper](#paper)
- [Abstract](#abstract)
- [Installation](#installation)
- [Install Dataset](#install-dataset)
- [Training](#training-from-scratch)
- [Sampling](#sampling)
- [Evaluation](#evaluation)
- [Citing](#citing)

## Paper

- **Paper:** [OpenReview](https://openreview.net/forum?id=bWtRZQ1rm2)
- **Poster:** [ICLR 2026 Poster](https://iclr.cc/media/PosterPDFs/ICLR%202026/10008602.png?t=1774447885.2973316)
- **Project Page:** [Link to project page (Coming soon)](#)

> *Poster and project page links will be added when available.*

## Note 📝

- ➡️ PRDiT architecture implemented [here](#) 📄
- ➡️ Trained models available [here](#) 💻
- ➡️ Training and evaluation code [here](#) ✨

## Updates 🎉

- *Add release milestones and updates here.*

## Abstract

<p align="center">
  <img src="assets/overview.png" width="95%" alt="PRDiT Architecture Overview">
</p>

Generating high-resolution 3D CT volumes with fine details remains challenging due to substantial computational demands and optimization difficulties inherent to existing generative models. In this paper, we propose the Pixel-Level Residual Diffusion Transformer (PRDiT), a scalable generative framework that synthesizes high-quality 3D medical volumes directly at voxel-level. PRDiT introduces a two-stage training architecture comprising 1) a local denoiser in the form of an MLP-based blind estimator operating on overlapping 3D patches to separate low-frequency structures efficiently, and 2) a global residual diffusion transformer employing memory-efficient attention to model and refine high-frequency residuals across entire volumes. This coarse-to-fine modeling strategy simplifies optimization, enhances training stability, and effectively preserves subtle structures without the limitations of an autoencoder bottleneck. Extensive experiments conducted on the LIDC-IDRI and RAD-ChestCT datasets demonstrate that PRDiT consistently outperforms state-of-the-art models, such as HA-GAN, 3D LDM and WDM-3D, achieving significantly lower 3D FID, MMD and Wasserstein distance scores.

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+

```bash
# Create conda environment
conda create -n prdit python=3.10
conda activate prdit

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Install Dataset

We use **LIDC-IDRI** and **RAD-ChestCT** for our experiments.

Detailed dataset download, preprocessing, and split-generation instructions are
available in [datasets/README.md](datasets/README.md).

## Training from Scratch

Use `--config {config_name}` to specify the config filename (e.g., `lidc.yaml`).

The subdirectory (`configs/local/` or `configs/global/`) is automatically selected:
`--from_scratch` resolves to `configs/local/{config_name}` (local denoiser);
omitting it resolves to `configs/global/{config_name}` (global residual PRDiT).

### Basic Training
```bash
# Single GPU
python train.py --config {config}

# Multi-GPU
torchrun --nproc_per_node=4 train.py --config {config}

# Debug mode
python train.py --config {config} --debug
```
### Progressive Training
```bash
# Stage 1: Train Local denoiser module (depth=0)
# Set model.name: "PRDiT-B/12/0" in config
python train.py --config {config} --from_scratch

# Stage 2: Train Global Residual PRDiT (depth>0)
# Set model.name: "PRDiT-B/12/4" in config
# Set pretrained_path: "/path/to/stage1/checkpoint.pt"
python train.py --config {config}
```

---

## Sampling

```
# Basic sampling
python sample.py --config {config} --ckpt $CKPT

# Custom parameters
python sample.py --config {config} --new --ckpt $CKPT --num-samples $SAMPLE_NUM --total-samples $STEP_NUM --output-dir $OUTPUT
```
**Output:** NIfTI files saved in specified directory.

## Evaluation

### Compute metrics (3D FID, MMD, Wasserstein distance)

The evaluation procedure runs as follows:

**3D FID Score**

```
python evaluations/fid.py --dataset $DATASET --img_size $IMG_SIZE --data_root_real $DATA_ROOT_REAL --data_root_fake $DATA_ROOT_FAKE --pretrain_path $PRETRAIN_PATH
```

**3D MMD Score**

```
python evaluations/mmd.py --dataset $DATASET --img_size $IMG_SIZE --data_root_real $DATA_ROOT_REAL --data_root_fake $DATA_ROOT_FAKE --pretrain_path $PRETRAIN_PATH
```

**WGAN Critic**

```bash
# Train for Wasserstein distance
python evaluations/wgan_gp.py --seed $SEED --save_path $SAVE_PATH --batch_size $BATCH_SIZE --img_size $IMG_SIZE --gpu_id $GPU_ID --dataset $DATASET --data_root_real $DATA_ROOT_REAL --data_root_fake_0 $DATA_ROOT_FAKE_0 --data_root_fake_1 $DATA_ROOT_FAKE_1 --train_size $TRAIN_SIZE --val_size $VAL_SIZE

# Evaluate for Wasserstein distance
python evaluations/wgan_gp.py --eval --seed $SEED --save_path $SAVE_PATH --batch_size $BATCH_SIZE --img_size $IMG_SIZE --gpu_id $GPU_ID --dataset $DATASET --data_root_real $DATA_ROOT_REAL --data_root_fake_0 $DATA_ROOT_FAKE_0 --data_root_fake_1 $DATA_ROOT_FAKE_1
```

## Citing

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{
zhang2026pixellevel,
title={Pixel-Level Residual Diffusion Transformer: Scalable 3D {CT} Volume Generation},
author={Zhenkai Zhang and Markus Hiller and Krista A. Ehinger and Tom Drummond},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=bWtRZQ1rm2}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
