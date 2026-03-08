# Water Pipe Leak Detection with Vision Transformers

> **Bachelor's Thesis · Jorge Ramírez Hernández**
> Leak detection in water pipelines using time-frequency representations and Vision Transformers (ViT).

---

## Overview

Complete pipeline for the thesis *"Leak Detection in Water Pipelines Using Vision Transformers"*.

The central idea is to convert **1-D acoustic signals** into **2-D time-frequency images**
(scalograms or spectrograms) and classify them with a **Vision Transformer (ViT)**.

```
Raw acoustic signal
        │
        ▼
  Wavelet Denoising  (db4, level 3)
        │
        ├──── CWT  →  Scalogram       (50 × 2048)
        │
        └──── STFT →  Log-Spectrogram (272 × 112)
                             │
                             ▼
                     Vision Transformer
                     ┌───────────────────┐
                     │  Patch Embedding  │
                     │  Positional Enc.  │
                     │  Transformer ×10  │
                     │  CLS → MLP Head   │
                     └───────────────────┘
                             │
                  ┌──────────┴──────────┐
              Binary               Multiclass
          (leak / no-leak)   (no-leak + 4 leak types)
```

---

## Dataset

Acoustic signals recorded in two pipe topologies at **25,600 Hz**.

| Topology | Description |
|----------|-------------|
| **Branched** | Pipe network with T-junctions |
| **Looped** | Closed-loop pipe network |

**Classes (multiclass task):**

| Label | Folder in dataset |
|-------|-------------------|
| 0 | Circumferential Crack |
| 1 | Gasket Leak |
| 2 | Longitudinal Crack |
| 3 | No-leak |
| 4 | Orifice Leak |

Source: [Mendeley Data — Water Pipe Leakage Detection](https://data.mendeley.com)

---

## Experiment Matrix

16 experiments covering all combinations of:

| Axis | Options |
|------|---------|
| Time-frequency transform | CWT · STFT |
| Pipe topology | Branched · Looped |
| Classification task | Binary · Multiclass |
| Model initialization | Scratch · Pretrained ImageNet (ViT-B/16) |

---

## Repository Structure

```
├── train.py                    # Single entry point for all 16 experiments
├── test_pipeline.py            # Tests with synthetic data (no real dataset required)
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # WaveletDenoising, denoise_signal_batch
│   │   ├── transforms.py      # CWT scalograms + STFT spectrograms
│   │   └── dataset.py         # Dataset, DataLoader factory, label maps
│   ├── models/
│   │   └── vit.py             # VisionTransformer (scratch) + PretrainedViT + build_model()
│   ├── training/
│   │   └── trainer.py         # Training loop, early stopping, checkpoints
│   └── utils/
│       └── metrics.py         # Confusion matrix, ROC, PR, classification report
│
├── configs/
│   └── default.yaml           # Default hyperparameters
│
├── docs/
│   └── guia_colab.md          # Step-by-step guide for running on Google Colab
│
├── requirements.txt
└── .gitignore
```

---

## Data Pipeline

The pipeline is designed to never load the full dataset into RAM at once,
making it stable in memory-constrained environments like Google Colab.

```
Mendeley dataset CSVs
        │
        ▼
load_signals_from_csv()          # Reads 1-second frames, balances classes
        │
        ▼
denoise_signal_batch()           # Wavelet denoising signal by signal
        │
        ▼
CWT / STFT                       # One image at a time
        │
        ▼
cache .h5 to Drive               # Written image by image (no RAM spike)
        │
        ▼
HDF5Dataset (lazy)               # Reads from .h5 on demand, batch by batch
        │
        ▼
DataLoader → ViT → train()
```

The `.h5` file is computed once and reused across all subsequent runs
of the same `transform + topology + task` combination.
Scratch and pretrained experiments share the same cache.

---

## Quick Start

### 1 · Install dependencies

```bash
sudo apt-get install libfftw3-single3   # required by fCWT
pip install -r requirements.txt
pip install fCWT
```

### 2 · Verify installation

```bash
python test_pipeline.py --full
# Should print: 9 passed | 0 failed
```

### 3 · Run an experiment

`train.py` is the single entry point for all 16 experiments:

```bash
# ViT from scratch | STFT | Branched | Binary
python train.py --transform stft --topology branched --task binary \
    --data_dir /path/to/Dataset --cache_dir /path/to/cache

# ViT from scratch | CWT | Looped | Multiclass
python train.py --transform cwt --topology looped --task multiclass \
    --data_dir /path/to/Dataset --cache_dir /path/to/cache

# Pretrained ViT (ImageNet) | STFT | Branched | Binary
python train.py --transform stft --topology branched --task binary --pretrained \
    --data_dir /path/to/Dataset --cache_dir /path/to/cache

# Override hyperparameters
python train.py --transform cwt --topology looped --task multiclass \
    --lr 5e-5 --epochs 100 --batch_size 16
```

Outputs saved to `checkpoints/{transform}_{topology}_{task}_{model}/`:

```
checkpoints/stft_branched_binary_scratch/
├── best_model.pt
├── training_history.png
├── confusion_matrix.png
├── roc_curve.png
└── pr_curve.png
```

### 4 · Google Colab

See [`docs/guia_colab.md`](docs/guia_colab.md) for step-by-step instructions,
including how to enable GPU, manage the Drive cache, and troubleshoot issues.

---

## Model Architecture

### ViT from Scratch

| Hyperparameter | Value |
|----------------|-------|
| Embedding dim | 768 |
| Depth (blocks) | 10 |
| Attention heads | 8 |
| MLP ratio | 4× |
| Dropout | 0.15 |
| Attention dropout | 0.05 |
| Optimizer | AdamW |
| LR scheduler | ReduceLROnPlateau |

### Pretrained ViT

- Backbone: **ViT-Base/16** (ImageNet-21k weights via `timm`)
- Images are resized to 224 × 224 with bilinear interpolation
- Single (grayscale) channel is replicated to 3 channels (RGB) before the backbone
- Only the classification head is replaced and trained from scratch

### Input Shapes by Transform

| Transform | Image shape | Patch size | No. patches |
|-----------|-------------|------------|-------------|
| CWT | 50 × 2048 | 5 × 16 | 10 × 128 = 1280 |
| STFT | 272 × 112 | 8 × 8 | 34 × 14 = 476 |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning |
| `timm` | Pretrained ViT-B/16 backbone |
| `PyWavelets` | Wavelet denoising + CWT fallback |
| `fCWT` | Fast CWT (GPU/CPU) |
| `scipy` | STFT |
| `h5py` | Image cache in HDF5 |
| `scikit-learn` | Metrics, stratified splits |
| `seaborn` | Confusion matrix visualization |

---

## Citation

```bibtex
@thesis{ramirez2025leak,
  author  = {Ramírez Hernández, Jorge},
  title   = {Leak Detection in Water Pipelines Using Vision Transformers},
  school  = {[University]},
  year    = {2025},
  type    = {Bachelor's Thesis}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
