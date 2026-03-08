"""
dataset.py
----------
PyTorch Dataset wrappers and data-loading utilities for the
water-pipe leak detection pipeline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

# Folder name → integer label  (used when reading CSVs)
MULTICLASS_LABEL_CODES: Dict[str, int] = {
    "Circumferential Crack": 0,
    "Gasket Leak":           1,
    "Longitudinal Crack":    2,
    "No-leak":               3,
    "Orifice Leak":          4,
}

BINARY_LABEL_CODES: Dict[str, int] = {
    "Leak":    0,
    "No-leak": 1,
}

# Integer label → human-readable name  (used for reports / plots)
MULTICLASS_LABELS: Dict[int, str] = {v: k for k, v in MULTICLASS_LABEL_CODES.items()}
BINARY_LABELS: Dict[int, str]     = {v: k for k, v in BINARY_LABEL_CODES.items()}

# Folder names that count as "leak" in binary mode
LEAK_CLASS_NAMES = {"Circumferential Crack", "Gasket Leak", "Longitudinal Crack", "Orifice Leak"}


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LeakDetectionDataset(Dataset):
    """
    PyTorch Dataset for time-frequency images (scalograms / spectrograms).

    Parameters
    ----------
    data : torch.Tensor  (N, C, H, W)
    labels : torch.Tensor  (N,) or (N, 1)
    transform : callable, optional
        torchvision-style transform applied per sample.
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# ---------------------------------------------------------------------------
# CSV loader  (mirrors load_accelerometer_data from the original notebooks)
# ---------------------------------------------------------------------------

def load_signals_from_csv(
    data_dir: str,
    task: str = "multiclass",
    sample_rate: int = 25600,
    downsample_factor: int = 1,
    fraction_to_include: float = 1.0,
    test_size: float = 0.2,
    seed: int = 53,
) -> Tuple[Dict, Dict]:
    """
    Load raw acoustic signals from the Mendeley CSV dataset.

    Expected folder layout::

        data_dir/                        (e.g. Dataset/Branched/)
        ├── Circumferential Crack/   ← one .csv per recording
        ├── Gasket Leak/
        ├── Longitudinal Crack/
        ├── No-leak/
        └── Orifice Leak/

    Each CSV must contain a column named ``Value``.

    Replicates the exact preprocessing from the original notebooks:

    * Downsampled by ``downsample_factor``.
    * Up to 30 seconds per file (``sample_rate * 30`` samples).
    * Segmented into non-overlapping 1-second frames.
    * **Binary mode** — class balancing so total_leak ≈ total_no_leak,
      with each of the 4 leak types contributing equally.

    Returns
    -------
    signals_dict : ``{'training': [np.ndarray, ...], 'testing': [...]}``
    labels_dict  : ``{'training': [int, ...],        'testing': [...]}``
    """
    import pandas as pd

    def _rm_ds(path):
        for n in [".DS_Store", ".DS_store"]:
            fp = os.path.join(path, n)
            if os.path.isfile(fp):
                os.remove(fp)

    _rm_ds(data_dir)

    # ── Collect frames ─────────────────────────────────────────────────────
    leak_type_signals: Dict[str, list] = {k: [] for k in LEAK_CLASS_NAMES}
    no_leak_signals: list = []
    all_signals: list = []
    all_labels:  list = []
    stratify_col: list = []

    for class_folder in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        _rm_ds(class_path)

        if class_folder not in MULTICLASS_LABEL_CODES:
            print(f"  [info] Carpeta desconocida, se omite: {class_folder}")
            continue

        n_loaded = 0
        for fname in sorted(os.listdir(class_path)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(class_path, fname)
            try:
                df  = pd.read_csv(fpath, index_col=False)
                df  = df.iloc[::downsample_factor, :].reset_index(drop=True)
                sig = df["Value"].values[: sample_rate * 30].astype(np.float32)
            except Exception as e:
                print(f"  [warning] No se pudo leer {fpath}: {e}")
                continue

            n_frames   = len(sig) // sample_rate
            frame_limit = int(fraction_to_include * n_frames)
            starts      = np.linspace(0, len(sig) - sample_rate, n_frames)

            for i, start in enumerate(starts):
                if i >= frame_limit:
                    break
                frame = sig[int(start): int(start) + sample_rate]
                if len(frame) != sample_rate:
                    continue

                if task == "multiclass":
                    all_signals.append(frame)
                    all_labels.append(MULTICLASS_LABEL_CODES[class_folder])
                    stratify_col.append(MULTICLASS_LABEL_CODES[class_folder])
                else:  # binary — store separately for balancing
                    if class_folder == "No-leak":
                        no_leak_signals.append(frame)
                    elif class_folder in LEAK_CLASS_NAMES:
                        leak_type_signals[class_folder].append(frame)
                n_loaded += 1

        print(f"  Clase '{class_folder}': {n_loaded} frames")

    # ── Binary balancing ───────────────────────────────────────────────────
    if task == "binary":
        total_no_leak    = len(no_leak_signals)
        min_available    = min(len(v) for v in leak_type_signals.values())
        samples_per_type = min(total_no_leak // 4, min_available)

        print(f"\n[data] Balanceo binario:")
        print(f"       No-leak frames : {total_no_leak}")
        print(f"       Leak por tipo  : {samples_per_type}  (×4 = {samples_per_type * 4})")

        all_signals  = list(no_leak_signals)
        all_labels   = [BINARY_LABEL_CODES["No-leak"]] * total_no_leak
        stratify_col = ["No-leak"] * total_no_leak

        for leak_type, frames in leak_type_signals.items():
            all_signals  += frames[:samples_per_type]
            all_labels   += [BINARY_LABEL_CODES["Leak"]] * samples_per_type
            stratify_col += [leak_type] * samples_per_type

    # ── Train / test split ─────────────────────────────────────────────────
    sig_train, sig_test, lbl_train, lbl_test = train_test_split(
        all_signals, all_labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_col,
    )

    print(f"\n[data] Total: {len(all_signals)} frames  |  "
          f"Train: {len(sig_train)}  |  Test: {len(sig_test)}")

    return (
        {"training": sig_train, "testing": sig_test},
        {"training": lbl_train, "testing": lbl_test},
    )


# ---------------------------------------------------------------------------
# HDF5 loaders
# ---------------------------------------------------------------------------

def load_signals_h5(filepath: str) -> Dict:
    """Load raw signals and labels from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        return {k: f[k][:] for k in f.keys()}


def load_images_h5(filepath: str) -> Dict:
    """Load pre-computed time-frequency images from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        return {k: f[k][:] for k in f.keys()}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    *,
    is_binary: bool,
    val_size: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build train and val DataLoaders from the training split.

    The test set is handled separately in train.py (it comes from
    load_signals_from_csv and is never mixed with the training data here).

    Parameters
    ----------
    x : np.ndarray  (N, 1, H, W) or (N, H, W)
    y : np.ndarray  (N,)
    is_binary : bool
    val_size : float
        Fraction of x carved out for validation.

    Returns
    -------
    dict with keys ``'train'`` and ``'val'``.
    """
    if x.ndim == 3:
        x = x[:, np.newaxis, :, :]  # (N, H, W) → (N, 1, H, W)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=val_size,
        random_state=seed,
        stratify=y,
    )

    def _tensors(xd, yd):
        xt = torch.tensor(xd, dtype=torch.float32)
        yt = (
            torch.tensor(yd, dtype=torch.float32).unsqueeze(1)
            if is_binary
            else torch.tensor(yd, dtype=torch.long)
        )
        return TensorDataset(xt, yt)

    pin = torch.cuda.is_available()
    return {
        "train": DataLoader(_tensors(x_train, y_train),
                            batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin),
        "val":   DataLoader(_tensors(x_val, y_val),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin),
    }
