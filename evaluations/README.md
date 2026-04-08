# Evaluation Metrics

This directory contains evaluation metrics for PRDiT:

| Script | Metric | What it measures |
|--------|--------|-----------------|
| `fid.py` | 3D FID | Feature-space distribution distance (lower is better) |
| `mmd.py` | 3D MMD | Maximum Mean Discrepancy (lower is better) |
| `ms_ssim.py` | MS-SSIM | Sample diversity across all generated pairs (higher is better) |

FID and MMD both use a pretrained **3D ResNet-50** ([MedicalNet](https://github.com/Tencent/MedicalNet)) as the feature extractor.

---

## Setup

Create and activate the dedicated evaluation environment:

```bash
conda env create -f evaluations/eval_env.yml
conda activate eval
```

All scripts must be run from the **project root** (not from inside `evaluations/`):

```bash
cd /path/to/
python evaluations/fid.py ...
```

---

## Pretrained Weights

FID and MMD require a pretrained 3D ResNet-50 checkpoint from [MedicalNet](https://github.com/Tencent/MedicalNet).

**Download steps:**

1. Go to the [MedicalNet releases](https://github.com/Tencent/MedicalNet/releases) page.
2. Download `MedicalNet_pytorch_files2.tar.gz`.
3. Extract and place the checkpoint:

```bash
tar -xzf MedicalNet_pytorch_files2.tar.gz
mkdir -p evaluations/pretrained
cp resnet_50.pth evaluations/pretrained/resnet_50.pth
```

The checkpoint must contain a `state_dict` key compatible with the 3D ResNet-50 in `evaluations/models/resnet.py`.

---

## 3D FID

Computes the Fréchet Inception Distance between real and generated volumes using 2048-D ResNet-50 features.

```bash
python evaluations/fid.py \
    --dataset lidc-idri \
    --img_size 128 \
    --data_root_real /path/to/real/volumes \
    --data_root_fake /path/to/generated/volumes \
    --pretrain_path evaluations/pretrained/resnet_50.pth \
    --path_to_activations /path/to/save/stats \
    --gpu_id 0
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `lidc-idri` or `rad_chestCT` |
| `--num_samples` | `1000` | Number of volumes used to compute FID |
| `--manual_seed` | `192` | Random seed for reproducibility |
| `--path_to_activations` | required | Directory where `mu_real.npy` / `sigma_real.npy` etc. are saved |

---

## 3D MMD

Computes Maximum Mean Discrepancy with an RBF kernel. Gamma is selected automatically via the median heuristic.

```bash
python evaluations/mmd.py \
    --dataset lidc-idri \
    --img_size 128 \
    --data_root_real /path/to/real/volumes \
    --data_root_fake /path/to/generated/volumes \
    --pretrain_path evaluations/pretrained/resnet_50.pth \
    --path_to_activations /path/to/save/stats \
    --gpu_id 0
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | `lidc-idri` or `rad_chestCT` |
| `--num_samples` | `1000` | Number of volumes used to compute MMD |
| `--manual_seed` | `1` | Random seed for reproducibility |
| `--path_to_activations` | required | Directory where activations are saved |

---

## MS-SSIM (Diversity)

Computes the Multi-Scale SSIM across all pairs of generated samples to measure diversity (lower MS-SSIM means higher diversity). Runs on GPU if available.

```bash
python evaluations/ms_ssim.py \
    --dataset lidc-idri \
    --img_size 128 \
    --sample_dir /path/to/generated/volumes \
    --num_workers 4
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample_dir` | required | Directory of generated NIfTI volumes |
| `--max_samples` | `None` (all) | Cap on number of samples (N² pairs evaluated) |
| `--seed` | `42` | Random seed |

> **Note:** MS-SSIM evaluates all N(N-1)/2 unique pairs, so runtime scales quadratically with `--max_samples`. Set this to a reasonable number for tractable compute.

---

## W-Critic (Wasserstein Distance)

The Wasserstein distance is estimated via a trained WGAN-GP critic, as proposed in the paper. The implementation is maintained in a separate repository:

> **W-Critic repo:** [coming soon](#) *(link will be updated)*

Once available, training and evaluation follow:

```bash
# Train the W-critic
python evaluations/wgan_gp.py \
    --dataset $DATASET --img_size $IMG_SIZE --gpu_id $GPU_ID \
    --data_root_real $DATA_ROOT_REAL \
    --data_root_fake_0 $DATA_ROOT_FAKE_0 \
    --data_root_fake_1 $DATA_ROOT_FAKE_1 \
    --train_size $TRAIN_SIZE --val_size $VAL_SIZE \
    --save_path $SAVE_PATH

# Evaluate Wasserstein distance
python evaluations/wgan_gp.py --eval \
    --dataset $DATASET --img_size $IMG_SIZE --gpu_id $GPU_ID \
    --data_root_real $DATA_ROOT_REAL \
    --data_root_fake_0 $DATA_ROOT_FAKE_0 \
    --data_root_fake_1 $DATA_ROOT_FAKE_1 \
    --save_path $SAVE_PATH
```

---

## Supported Datasets

| `--dataset` value | Dataset |
|-------------------|---------|
| `lidc-idri` | LIDC-IDRI lung CT |
| `rad_chestCT` | RAD-ChestCT whole-chest CT |

See [datasets/README.md](../datasets/README.md) for preprocessing and split instructions.
