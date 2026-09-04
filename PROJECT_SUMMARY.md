# Project Module Summary: BiDA (Bi-directional Domain Adaptation for Cross-domain HSI Classification)

## Overview

This is a PyTorch implementation of **Cross-Domain Hyperspectral Image Classification Based on Bi-Directional Domain Adaptation (BiDA)** for cross-temporal and cross-scene hyperspectral image classification, published in IEEE TCSVT 2025.

**Default experiment:** Houston13 → Houston18 (cross-temporal domain adaptation)

---

## Directory Structure

```
IEEE_TCSVT_BiDA/
├── main.py                  # Entry point / CLI argument parser
├── train_pipeline.py        # Training, validation, testing loops
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── get_model.py         # Model factory
│   ├── BiDA.py              # Proposed BiDA model
│   ├── m3ddcnn.py           # Baseline: M3D-DCNN
│   ├── cnn3d.py             # Baseline: 3-D CNN
│   ├── rssan.py             # Baseline: RSSAN
│   ├── ablstm.py            # Baseline: AB-LSTM
│   ├── dffn.py              # Baseline: DFFN
│   ├── speformer.py         # Baseline: SpectralFormer
│   ├── ssftt.py             # Baseline: SSFTT
│   └── GAHT.py              # Baseline: GAHT
├── loss/                    # Loss functions
│   ├── __init__.py
│   ├── make_loss.py         # Loss factory
│   ├── softmax_loss.py      # Cross-entropy + label smoothing
│   ├── center_loss.py       # Center loss
│   ├── triplet_loss.py      # Triplet loss
│   ├── mmd_loss.py          # MMD loss
│   ├── lmmd_loss.py         # Local MMD loss
│   ├── arcface.py           # ArcFace / CircleLoss / Cosface / AMSoftmax
│   └── metric_learning.py   # Contrastive loss
├── utils/                   # Utilities
│   ├── dataset.py           # Dataset loading, sampling, augmentation
│   ├── utils_HSI.py         # HSI evaluation metrics, sliding window, visualization
│   └── scheduler.py         # Optimizer & LR scheduler factory
└── Houston/                 # Default dataset folder
```

---

## Modules

### 1. `main.py` — Entry Point & Experiment Orchestration

| Attribute | Detail |
|-----------|--------|
| **Purpose** | CLI argument parsing, dataset loading, train/val/test split, model initialization, training launch |
| **Inputs** | CLI args: `--model`, `--source_name`, `--target_name`, `--dataset_dir`, `--patch_size`, `--epoch`, `--bs`, `--lr`, `--ratio`, `--dim`, `--depth`, `--num_tokens`, `--loss_type`, `--lambda1`, `--lambda2`, `--ema_decay`, `--seed`, `--re_ratio`, `--device`, `--num_workers`, `--log_interval`, `--labelsmooth` |
| **Outputs** | Trained model checkpoints (`model_ts_best*.pth`), result `.mat` files |
| **Parameters** | N/A (script-level orchestration) |

**Key operations:**
- Loads source and target HSI `.mat` files via `load_mat_hsi`
- Samples training/validation splits via `sample_gt` with stratification
- Pads HSI cubes with reflective padding based on `patch_size`
- Creates `HSIDataset` and `DataLoader` for train, val, and test
- Instantiates both main model and EMA (Exponential Moving Average) model via `get_model`
- Loads optimizer/scheduler via `load_scheduler`
- Loads loss function via `make_loss`
- Calls `train()` from `train_pipeline.py`

---

### 2. `train_pipeline.py` — Training, Validation & Inference

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Core training loop with domain adaptation losses, validation, sliding-window inference |
| **Inputs** | `network`, `network_ema`, `optimizer`, `criterion`, `num_classes`, `train_loader`, `val_loader`, `test_loader_noise`, `test_loader`, `opts`, `saving_path`, `device`, `scheduler` |
| **Outputs** | Best model checkpoint, `.mat` results file |
| **Parameters** | N/A (orchestration module) |

**Key functions:**

- `train()`: Iterates over epochs. For each batch from source (`train_loader`) and target (`test_loader_noise`):
  - Computes classification loss (`criterion(out_x, targets)`)
  - Computes bidirectional distillation losses (`distill_loss`)
  - Computes EMA consistency losses (`softmax_mse_loss`)
  - After epoch 100, adds MMD alignment loss
  - Updates EMA model variables
  - Runs periodic validation and saves best checkpoint

- `validation()`: Evaluates model on validation loader, returns OA (Overall Accuracy) and metrics

- `test()`: Full-image inference using sliding window, returns per-pixel class probability map

- `update_ema_variables()`: Exponential Moving Average update for teacher model

- `distill_loss()`: KL divergence between teacher and student softmax outputs

- `softmax_mse_loss()`: MSE between softmax probabilities of two models

---

### 3. `models/get_model.py` — Model Factory

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Instantiates the requested model architecture by name; optionally creates EMA copy |
| **Inputs** | `model_name` (str), `dataset_name` (str), `patch_size` (int), `opts` (Namespace), `ema` (bool) |
| **Outputs** | Instantiated `nn.Module` |
| **Parameters** | N/A |

**Supported models:** `m3ddcnn`, `cnn3d`, `rssan`, `ablstm`, `dffn`, `speformer`, `GAHT`, `ssftt`, `BiDA`

---

### 4. `models/BiDA.py` — Proposed BiDA Architecture

| Attribute | Detail |
|-----------|--------|
| **Purpose** | The proposed Bi-directional Domain Adaptation network with triple-branch attention transformer |
| **Inputs** | `x` (source HSI patch: `[B, 1, n_bands, H, W]`), `x_tar` (target HSI patch: same shape) |
| **Outputs** | Training: `(out_x, out_x_tar, out_x_fusion, out_fusion_src)` — 4 logit tensors `[B, num_classes]`. Inference: `(None, out_x_tar, None)` or `(None, out_x_tar, None, feat)` |
| **Parameters** | ~376K (Houston) to ~967K (Dioni/Loukia), configurable via `opts.dim`, `opts.depth`, `opts.num_tokens` |

**Architecture components:**

| Component | Purpose | Parameters |
|-----------|---------|------------|
| `conv3d_features` | 3D spectral-spatial convolution (1→8 channels, kernel 3×3×3) | ~224 |
| `conv2d_features` | 2D spatial convolution (8×n_bands → dim, kernel 3×3) | ~(8×bands+1)×dim×9 + dim |
| `_forward_semantic_tokens()` | Generates `num_tokens` semantic tokens via spatial attention | ~(dim×L + 64×L + 64×dim) |
| `pos_embedding` | Learnable positional embedding `[1, L+1, dim]` | (L+1)×dim |
| `cls_token` | Classification token `[1, 1, dim]` | dim |
| `blocks` (×depth) | Triple-branch transformer blocks with bidirectional attention | ~depth × (4×dim² + …) |
| `nn1` | Final linear classifier `dim → num_classes` | dim × num_classes |

**Key classes:**
- `DropPath`: Stochastic depth regularization
- `Mlp`: Feed-forward network with GELU activation
- `Attention_triple_branches`: Multi-head attention computing source self-attn, target self-attn, and cross-domain attention (source-query + target-key/value)
- `Block_triple_branches`: Transformer block with 4 output branches (source, target, target→source fusion, source→target fusion)

**Dataset-specific configs:**
- Houston13/18: `n_bands=48`, `num_classes=7`
- Dioni/Loukia: `n_bands=176`, `num_classes=12`
- MJG datasets: `n_bands=64`, `num_classes=5`

---

### 5. `models/m3ddcnn.py` — M3D-DCNN Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Multi-scale 3D deep CNN with Inception-style multi-kernel spectral branches |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 1,627,872 · pu: 432,665 · whulk: 1,239,065 · hrl: 1,224,670 |

**Architecture:**
- `conv1`: Conv3d 1→16, kernel (11,3,3), stride (3,1,1)
- `conv2_x` (4 branches): Multi-scale spectral convs with kernel sizes 1, 3, 5, 11 (summed)
- `conv3_x` (4 branches): Second multi-scale stage
- `conv4`: Conv3d 16→16, kernel (3,2,2)
- `pooling`: MaxPool3d (3,2,2)
- `fc`: Linear → num_classes (dropout 0.6)

---

### 6. `models/cnn3d.py` — 3-D CNN Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Standard 3-D CNN for HSI classification (Hamida et al., 2018) |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 1,209,996 · pu: 387,839 · whulk: 898,139 · hrl: 943,504 |

**Architecture:**
- `conv1`: Conv3d 1→20, kernel (3,3,3)
- `pool1`: Conv3d 20→20, kernel (3,1,1), stride (2,1,1) — spectral downsampling
- `conv2`: Conv3d 20→35, kernel (3,3,3)
- `pool2`: Conv3d 35→35, kernel (3,1,1), stride (2,1,1)
- `conv3`: Conv3d 35→35, kernel (3,1,1)
- `conv4`: Conv3d 35→35, kernel (2,1,1), stride (2,1,1)
- `fc`: Linear → num_classes (dropout 0.5)

---

### 7. `models/rssan.py` — RSSAN Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Residual Spectral-Spatial Attention Network (Zhu et al., 2021) |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 109,840 · pu: 71,720 · whulk: 135,352 · hrl: 98,949 |

**Architecture components:**

| Component | Purpose |
|-----------|---------|
| `SSA` | Spectral-Spatial Attention module (SpectralAttention + SpatialAttention) |
| `SpectralAttention` | Channel-wise attention via AvgPool + MaxPool + Shared MLP |
| `SpatialAttention` | Spatial attention via channel-wise AvgPool + MaxPool + Conv2d |
| `RSSA` | Residual block with spectral-spatial attention |
| `block1` | Initial Conv2d + BN + ReLU |
| `fc` | Linear(patch_size², num_classes) |

---

### 8. `models/ablstm.py` — AB-LSTM Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Attention-based Bidirectional LSTM for HSI classification (Mei et al., 2021) |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 1,635,720 · pu: 517,656 · whulk: 2,761,635 · hrl: 1,252,170 |

**Architecture (Sequential):**

| Component | Purpose |
|-----------|---------|
| `SpatialAttention` | 2D CNN with skip connection extracting center-pixel context |
| `SpectralAttention` | 1D Conv + residual block for spectral feature refinement |
| `BiLSTM` | Bidirectional LSTM (hidden=64, layers=2, dropout=0.5) → Linear classifier |

---

### 9. `models/dffn.py` — DFFN Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Deep Feature Fusion Network with multi-stage residual blocks (Song et al., 2018) |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 423,360 · pu: 505,801 · whulk: 529,849 · hrl: 516,638 |

**Architecture:**

| Component | Purpose |
|-----------|---------|
| `basic_block` | 2× Conv2d residual block (3×3) |
| `trans_block` | Transition block between stages (1×1 + 3×3 convs) |
| `stage1/2/3` | Sequences of basic blocks with increasing filters (16→32→64) |
| `conv_stage1/2` | 3×3 convs for multi-scale feature fusion |
| `avgpool` + `fc` | Global average pooling → Linear classifier |

---

### 10. `models/speformer.py` — SpectralFormer Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | SpectralFormer with spectral-aware patch embedding and ViT/CAF transformer (Hong et al., 2022) |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 388,629 · pu: 194,153 · whulk: 581,092 · hrl: 322,447 |

**Architecture:**

| Component | Purpose |
|-----------|---------|
| `gain_neighborhood_band()` | Extracts neighboring spectral bands for each pixel |
| `ViT` | Vision Transformer with cls_token, pos_embedding, patch embedding |
| `Transformer` | Multi-head attention + FFN blocks with CAF (Cross-scale Attention Fusion) skip connections |
| `mlp_head` | LayerNorm → Linear(num_classes) |

**Config:** `dim=64`, `depth=5`, `heads=4`, `mlp_dim=8`, `near_band=3`, `mode="CAF"`

---

### 11. `models/ssftt.py` — SSFTT Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Spectral-Spatial Feature Tokenization Transformer (Sun et al., 2022) |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 955,016 · pu: 489,153 · whulk: 1,258,689 · hrl: 825,862 |

**Architecture:**

| Component | Purpose |
|-----------|---------|
| `conv3d_features` | 3D conv (1→8, kernel 3×3×3) |
| `conv2d_features` | 2D conv (8×(n_bands-2) → 64, kernel 3×3) |
| `token_wA`, `token_wV` | Learnable tokenization matrices (affinity + value) |
| `pos_embedding` | Positional embedding `[1, L+1, dim]` |
| `cls_token` | Classification token |
| `Transformer` | Standard Transformer encoder (PreNorm + Attention + MLP) |
| `nn1` | Linear classifier |

---

### 12. `models/GAHT.py` — GAHT Baseline

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Grouped Attention HSI Transformer with multi-stage grouped pixel embedding |
| **Inputs** | `[B, 1, n_bands, H, W]` |
| **Outputs** | `[B, num_classes]` logits |
| **Parameters** | sa: 972,624 · pu: 927,113 · whulk: 1,514,121 · hrl: 816,846 |

**Architecture:**

| Component | Purpose |
|-----------|---------|
| `pad` | ReplicationPad3d to align band count to group size |
| `GroupedPixelEmbedding` | Grouped Conv2d (groups=n_groups) + BN + ReLU → token sequence |
| `Block` | Standard Transformer block (PreNorm + Attention + MLP) |
| `MyTransformer` | Multi-stage (3 stages) hierarchical transformer with progressive dimension reduction |
| `head` | Mean pooling → Linear(num_classes) |

---

### 13. `utils/dataset.py` — Dataset Loading & Sampling

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Load HSI `.mat`/`.tif` files, sample train/val/test splits, apply data augmentation |
| **Inputs** | Dataset name, directory path, normalization method |
| **Outputs** | `(img, gt, labels)` — normalized HSI cube, ground truth map, class names |

**Key functions/classes:**

| Name | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| `open_file()` | Load `.mat` (via `scipy`/`h5py`) or `.tif` files | file path | dict / ndarray |
| `load_mat_hsi()` | Load named HSI dataset with normalization | `dataset_name`, `dataset_dir`, `norm` | `img` (H×W×bands), `gt` (H×W), `labels` |
| `sample_gt()` | Stratified random sampling of train/test masks | `gt`, `train_size`, `seed`, `mode` | `train_gt`, `test_gt` |
| `HSIDataset` | PyTorch Dataset with patch extraction and augmentation | `image`, `gt`, `patch_size`, augmentation flags | `(data, label)` tensors |

**Supported datasets:** `sa`, `pu`, `whulk`, `hrl`, `Loukia`, `Dioni`, `Houston18`, `Houston13`

**Normalization modes:** `normband`, `minmax`, `std`, `ln`, `sklearn`, `ori`

---

### 14. `utils/utils_HSI.py` — HSI Evaluation Utilities

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Metrics computation, sliding window inference helpers, visualization |
| **Inputs** | Predictions, ground truth, class count |
| **Outputs** | OA, Kappa, F1, confusion matrix |

**Key functions:**

| Name | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| `seed_worker()` | Set all random seeds for reproducibility | `seed` | N/A |
| `sliding_window()` | Generate sliding window patches over HSI | `image`, `step`, `window_size` | `(patch, x, y, w, h)` tuples |
| `count_sliding_window()` | Count total sliding windows | same | `int` |
| `grouper()` | Batch sliding window patches | `n`, `iterable` | batched chunks |
| `metrics()` | Compute OA, Kappa, F1, confusion matrix | `prediction`, `target`, `n_classes` | dict |
| `show_results()` | Print/visualize results | `results`, `vis` | formatted text |
| `compute_imf_weights()` | Inverse median frequency class weights | `ground_truth` | weight array |

---

### 15. `utils/scheduler.py` — Optimizer & Scheduler Factory

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Configure model-specific optimizers and learning rate schedulers |
| **Inputs** | `model_name`, `model`, `opts` |
| **Outputs** | `(optimizer, scheduler)` |

**Per-model configurations:**

| Model | Optimizer | LR | Scheduler |
|-------|-----------|-----|-----------|
| `m3ddcnn` | Adagrad | 0.01 | None |
| `cnn3d` | SGD + momentum | 0.001 | MultiStepLR([100,200], γ=0.1) |
| `rssan` | RMSprop | 0.001 | None |
| `ablstm` | Adam | 0.0005 | MultiStepLR([150], γ=0.1) |
| `dffn` | SGD + momentum | 0.1 | MultiStepLR([100,200], γ=0.1) |
| `speformer` | Adam | 1e-3 | MultiStepLR([30,60,...270], γ=0.9) |
| `ssftt` / `BiDA` | SGD | `opts.lr` (default 1e-2) | None |
| `GAHT` | SGD + momentum | 0.001 | None |

---

### 16. `loss/make_loss.py` — Loss Factory

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Instantiate loss function and center criterion based on config |
| **Inputs** | `cfg` (Namespace), `num_classes` |
| **Outputs** | `(loss_func, center_criterion)` |

**Supported loss types:**

| `loss_type` | Description |
|-------------|-------------|
| `softmax` | Cross-entropy (with optional label smoothing) |
| `softmax_center` | Cross-entropy + Center loss (weighted) |

---

### 17. Loss Functions

#### 17.1 `loss/softmax_loss.py` — CrossEntropyLabelSmooth

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Cross-entropy with label smoothing regularization |
| **Inputs** | `inputs` `[B, num_classes]`, `targets` `[B]` |
| **Outputs** | Scalar loss |
| **Parameters** | 0 (no learnable params) |

#### 17.2 `loss/center_loss.py` — CenterLoss

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Minimize intra-class feature distance to class centers |
| **Inputs** | `x` `[B, feat_dim]`, `labels` `[B]` |
| **Outputs** | Scalar loss |
| **Parameters** | `num_classes × feat_dim` (default 14,336 for 7 classes × 2048 dim) |

#### 17.3 `loss/triplet_loss.py` — TripletLoss

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Hard example mining triplet loss |
| **Inputs** | `global_feat` `[B, D]`, `labels` `[B]` |
| **Outputs** | `(loss, dist_ap, dist_an)` |
| **Parameters** | 0 (no learnable params) |

#### 17.4 `loss/mmd_loss.py` — MMD_loss

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Maximum Mean Discrepancy for domain alignment |
| **Inputs** | `source` `[B, D]`, `target` `[B, D]` |
| **Outputs** | Scalar MMD loss |
| **Parameters** | 0 (no learnable params) |

**Config:** `kernel_num=5`, `kernel_mul=2.0`, Gaussian RBF kernel

#### 17.5 `loss/lmmd_loss.py` — LMMDLoss (Local MMD)

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Local MMD with class-wise dynamic weighting and pseudo-labels |
| **Inputs** | `source`, `target`, `source_label`, `target_logits`, `device` |
| **Outputs** | Scalar LMMD loss |
| **Parameters** | 0 (no learnable params) |

#### 17.6 `loss/arcface.py` — ArcFace / CircleLoss / Cosface / AMSoftmax

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Large-margin classification losses for discriminative feature learning |
| **Inputs** | `input` `[B, in_features]`, `label` `[B]` |
| **Outputs** | `[B, out_features]` logits |
| **Parameters** | `out_features × in_features` (default 14,336 for 7 × 2048) |

#### 17.7 `loss/metric_learning.py` — ContrastiveLoss

| Attribute | Detail |
|-----------|--------|
| **Purpose** | Contrastive loss with margin-based positive/negative pairs |
| **Inputs** | `inputs` `[B, D]`, `targets` `[B]` |
| **Outputs** | Scalar loss |
| **Parameters** | 0 (no learnable params) |

---

## Training Pipeline Loss Composition

The training loop in `train_pipeline.py` combines losses as follows:

```
Total Loss = loss_cls 
           + λ₁ × (Align_loss + Distill_loss_src + Distill_loss_tar) 
           + λ₂ × (Consistency_loss_src + Consistency_loss_tar)
```

Where:
- **loss_cls** = Cross-entropy on source predictions
- **Align_loss** = MMD(source_feat, target_feat) + MMD(fusion_src, fusion_src) (added after epoch 100)
- **Distill_loss** = KL divergence between fusion output and counterpart branch output
- **Consistency_loss** = MSE between softmax of current and EMA model outputs
- **λ₁** (default 1e-1) weights alignment + distillation
- **λ₂** (default 1e+0) weights consistency

---

## Model Parameter Counts Summary

| Model | Houston13/18 | SA (16 cls) | PU (9 cls) | WHULK (9 cls) | HRL (14 cls) | Dioni/Loukia (12 cls) |
|-------|-------------|-------------|------------|---------------|--------------|----------------------|
| **BiDA** | **376,567** | N/A | N/A | N/A | N/A | **966,716** |
| **M3D-DCNN** | N/A | 1,627,872 | 432,665 | 1,239,065 | 1,224,670 | N/A |
| **3-D CNN** | N/A | 1,209,996 | 387,839 | 898,139 | 943,504 | N/A |
| **RSSAN** | N/A | 109,840 | 71,720 | 135,352 | 98,949 | N/A |
| **AB-LSTM** | N/A | 1,635,720 | 517,656 | 2,761,635 | 1,252,170 | N/A |
| **DFFN** | N/A | 423,360 | 505,801 | 529,849 | 516,638 | N/A |
| **SpectralFormer** | N/A | 388,629 | 194,153 | 581,092 | 322,447 | N/A |
| **SSFTT** | N/A | 955,016 | 489,153 | 1,258,689 | 825,862 | N/A |
| **GAHT** | N/A | 972,624 | 927,113 | 1,514,121 | 816,846 | N/A |

*Note: BiDA only supports Houston13/18 and Dioni/Loukia datasets. Other models do not support Houston datasets.*

---

## Data Flow Summary

```
main.py
  ├── load_mat_hsi()        → Raw HSI cube + GT + labels
  ├── sample_gt()           → Train/val/test masks
  ├── HSIDataset            → Patch extraction + augmentation
  ├── get_model()           → Main model + EMA model
  ├── load_scheduler()      → Optimizer + LR scheduler
  ├── make_loss()           → Classification loss + center criterion
  │
  └── train()
        ├── network(images_src, images_tar) → 4 logits (src, tar, fusion, fusion_src)
        ├── network_ema(images_src, images_tar) → 2 logits (no grad)
        ├── criterion(out_x, targets) → Classification loss
        ├── distill_loss() → Bidirectional distillation
        ├── softmax_mse_loss() → EMA consistency
        ├── MMD_loss() → Domain alignment (after epoch 100)
        │
        ├── validation() → OA, Confusion Matrix, F1, Kappa
        └── test() → Full-image sliding window inference
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 13 | Spatial patch size (odd) |
| `epoch` | 200 | Total training epochs |
| `bs` | 128 | Batch size |
| `lr` | 1e-2 | Learning rate |
| `ratio` | 0.95 | Source train/validation split ratio |
| `dim` | 64 | Transformer embedding dimension |
| `depth` | 3 | Number of transformer blocks |
| `num_tokens` | 4 | Number of semantic tokens |
| `lambda1` | 1e-1 | Weight for alignment + distillation |
| `lambda2` | 1e+0 | Weight for consistency loss |
| `ema_decay` | 0.999 | EMA decay factor |
| `loss_type` | `softmax` | Loss function type |
| `labelsmooth` | `off` | Enable/disable label smoothing |
| `seed` | 2100 | Random seed |
