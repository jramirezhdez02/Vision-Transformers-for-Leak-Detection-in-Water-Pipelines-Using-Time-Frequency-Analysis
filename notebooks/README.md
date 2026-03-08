# Notebooks

Each notebook is a self-contained Google Colab experiment that covers the full
pipeline for one combination of design choices.

## Naming convention

```
{Transform}_{Topology}_{Task}.ipynb
```

| Axis | Values |
|------|--------|
| **Transform** | `CWT` — Continuous Wavelet Transform (scalogram) · `STFT` — Short-Time Fourier Transform (log-spectrogram) |
| **Topology** | `Branched` — pipe network with branches · `Looped` — closed-loop pipe network |
| **Task** | `Binary` — leak vs. no-leak · `Multiclass` — no-leak + 4 pressure levels |

## Two model families

| Folder | Description |
|--------|-------------|
| `scratch_from_scratch/` | ViT trained **from scratch** with custom architecture and AdamW |
| `pretrained/` | ViT-Base/16 fine-tuned from **ImageNet weights** via `timm` |

## Full experiment matrix (16 notebooks)

|  | CWT Branched | CWT Looped | STFT Branched | STFT Looped |
|---|---|---|---|---|
| **Scratch Binary** | ✅ | ✅ | ✅ | ✅ |
| **Scratch Multiclass** | ✅ | ✅ | ✅ | ✅ |
| **Pretrained Binary** | ✅ | ✅ | ✅ | ✅ |
| **Pretrained Multiclass** | ✅ | ✅ | ✅ | ✅ |

## Running a notebook

1. Open in Google Colab (recommended — GPU required for CWT via fCWT).
2. Mount your Google Drive and set `data_dir` to the folder containing the
   Mendeley dataset.
3. Run cells top-to-bottom. All dependencies are installed in the first cell.

## Mendeley dataset

The raw acoustic signals come from the public Mendeley dataset:

> *Water Pipe Leakage Detection Using Acoustic Signals* — available at
> [https://data.mendeley.com](https://data.mendeley.com)

Folder structure expected in Drive:

```
MyDrive/
└── Mendeley_Data/
    └── Dataset/
        ├── Branched/
        │   ├── no_leak/
        │   └── leak_*/
        └── Looped/
            ├── no_leak/
            └── leak_*/
```
