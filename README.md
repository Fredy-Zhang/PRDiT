# PRDiT: Pixel-Level Residual Diffusion Transformer for Scalable 3D CT Volume Generation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **PRDiT** — *Pixel-Level Residual Diffusion Transformer* — a scalable approach for 3D CT volume generation, accepted at **ICLR 2026**.

## Paper

- **Paper:** [OpenReview](https://openreview.net/forum?id=bWtRZQ1rm2)
- **Poster:** [Link to poster (Coming soon)](#)
- **Project Page:** [Link to project page (Coming soon)](#)

> *Poster and project page links will be added when available.*

## Note 📝

- ➡️ PRDiT architecture implemented [here](#) 📄
- ➡️ Trained models available [here](#) 💻
- ➡️ Training and evaluation code [here](#) ✨

## Updates 🎉

- 

## Abstract

Generating high-resolution 3D CT volumes with fine details remains challenging due to substantial computational demands and optimization difficulties inherent to existing generative models. In this paper, we propose the Pixel-Level Residual Diffusion Transformer (PRDiT), a scalable generative framework that synthesizes high-quality 3D medical volumes directly at voxel-level. PRDiT introduces a two-stage training architecture comprising 1) a local denoiser in the form of an MLP-based blind estimator operating on overlapping 3D patches to separate low-frequency structures efficiently, and 2) a global residual diffusion transformer employing memory-efficient attention to model and refine high-frequency residuals across entire volumes. This coarse-to-fine modeling strategy simplifies optimization, enhances training stability, and effectively preserves subtle structures without the limitations of an autoencoder bottleneck. Extensive experiments conducted on the LIDC-IDRI and RAD-ChestCT datasets demonstrate that PRDiT consistently outperforms state-of-the-art models, such as HA-GAN, 3D LDM and WDM-3D, achieving significantly lower 3D FID, MMD and Wasserstein distance scores.

## Installation

Create a conda environment and install dependencies:

```bash
conda create -n prdit python=3.10
conda activate prdit
pip install -r requirements.txt
```

---

## Install Dataset

We use **LIDC-IDRI** and **RAD-ChestCT** for our experiments. Prepare the datasets as follows:

### LIDC-IDRI

1. Download the LIDC-IDRI dataset from [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
2. Place the data in `data/LIDC-IDRI/` (or update the path in your config).
3. Preprocess the CT scans:

### RAD-ChestCT

1. Download RAD-ChestCT from the official source [Zenodo](https://zenodo.org/records/6406114#.Ytl6OXbMLAQ).
2. Place the data in `data/RAD-ChestCT/` and run preprocessing:

---

## Training from Scratch

---

## Evaluation

### Generate samples

### Compute metrics (3D FID, MMD, Wasserstein distance)

---

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
