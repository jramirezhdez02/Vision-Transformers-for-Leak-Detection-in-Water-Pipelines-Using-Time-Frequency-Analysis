"""
train.py
--------
Single entry point to run any of the 16 experiments from the command line.

Usage
-----
python train.py --transform cwt --topology branched --task binary
python train.py --transform stft --topology looped --task multiclass --pretrained
python train.py --transform cwt --topology branched --task binary --lr 5e-5 --epochs 100

Run `python train.py --help` for the full list of options.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import (
    BINARY_LABELS,
    MULTICLASS_LABELS,
    load_signals_from_csv,
)
from src.data.preprocessing import denoise_signal_batch
from src.data.transforms import (
    calculate_scalograms_with_padding_modes,
    generate_log_spectrograms,
)
from src.models.vit import build_model
from src.training.trainer import set_seed, train
from src.utils.metrics import (
    get_predictions,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curve,
    plot_training_history,
    print_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5Dataset — reads images from disk batch by batch, never loads all at once
# ─────────────────────────────────────────────────────────────────────────────

class HDF5Dataset(Dataset):
    """
    Lazy-loading Dataset backed by an HDF5 file.

    The file is opened once per worker and images are read on demand,
    so RAM usage is proportional to batch_size, not to the full dataset.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 cache file.
    split : 'train' | 'val' | 'test'
        Which split to expose.  The file must contain datasets
        ``x_train`` / ``x_test`` and ``y_train`` / ``y_test``.
        Val indices are passed separately via ``indices``.
    is_binary : bool
    indices : array-like, optional
        Subset of row indices to expose (used to carve out a val split
        from the training portion without duplicating data on disk).
    """

    def __init__(self, h5_path: str, key_x: str, key_y: str,
                 is_binary: bool, indices=None):
        self.h5_path  = h5_path
        self.key_x    = key_x
        self.key_y    = key_y
        self.is_binary = is_binary
        self.indices  = indices  # None → use all rows

        # Read length without loading data
        with h5py.File(h5_path, "r") as f:
            self._len = len(f[key_y])
            if indices is None:
                self.indices = np.arange(self._len)

        self._file = None   # opened lazily per worker

    def _open(self):
        if self._file is None:
            # Simple read mode — compatible with Google Drive and local storage
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        row = int(self.indices[idx])
        x = torch.tensor(self._file[self.key_x][row], dtype=torch.float32)
        y_raw = int(self._file[self.key_y][row])

        if self.is_binary:
            y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(0)
        else:
            y = torch.tensor(y_raw, dtype=torch.long)
        return x, y


def make_h5_loaders(h5_path: str, is_binary: bool,
                    val_size: float, batch_size: int,
                    num_workers: int, seed: int):
    """
    Build train / val / test DataLoaders that read lazily from an HDF5 file.
    Never loads the full array into RAM.

    Note: if the H5 file is on Google Drive, num_workers is forced to 0
    because Drive does not support concurrent file access from multiple
    processes (DataLoader workers).  Local paths keep the requested value.
    """
    from sklearn.model_selection import train_test_split

    pin = torch.cuda.is_available()

    # Google Drive does not support multi-process H5 reads
    if "/content/drive" in h5_path or "MyDrive" in h5_path:
        if num_workers > 0:
            print(f"[data] Cache en Google Drive → num_workers forzado a 0 "
                  f"(Drive no soporta acceso H5 multi-proceso)")
        num_workers = 0

    # ── Train / val split on indices only (labels only — tiny) ────────────
    with h5py.File(h5_path, "r") as f:
        n_train     = len(f["y_train"])
        y_train_all = f["y_train"][:]

    idx_all = np.arange(n_train)
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=val_size,
        random_state=seed, stratify=y_train_all
    )

    ds_train = HDF5Dataset(h5_path, "x_train", "y_train", is_binary, idx_tr)
    ds_val   = HDF5Dataset(h5_path, "x_train", "y_train", is_binary, idx_val)
    ds_test  = HDF5Dataset(h5_path, "x_test",  "y_test",  is_binary, None)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin, persistent_workers=(num_workers > 0))

    return {
        "train": DataLoader(ds_train, shuffle=True,  **loader_kw),
        "val":   DataLoader(ds_val,   shuffle=False, **loader_kw),
        "test":  DataLoader(ds_test,  shuffle=False, **loader_kw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Water-pipe leak detection — ViT training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transform",  choices=["cwt", "stft"],         default="cwt")
    p.add_argument("--topology",   choices=["branched", "looped"],   default="branched")
    p.add_argument("--task",       choices=["binary", "multiclass"], default="binary")
    p.add_argument("--pretrained", action="store_true")

    p.add_argument("--data_dir",  default="data/raw",
                   help="Raíz del dataset de Mendeley (contiene Branched/ y Looped/).")
    p.add_argument("--cache_dir", default="data/processed",
                   help="Carpeta donde guardar/cargar los .h5 de imágenes.")
    p.add_argument("--no_cache",  action="store_true",
                   help="Forzar recálculo aunque exista caché.")

    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--val_split",    type=float, default=0.10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=2)

    p.add_argument("--out_dir",   default="checkpoints")
    p.add_argument("--no_plots",  action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def experiment_name(args) -> str:
    tag = "pretrained" if args.pretrained else "scratch"
    return f"{args.transform}_{args.topology}_{args.task}_{tag}"


def resolve_data_path(args) -> Path:
    folder = "Branched" if args.topology == "branched" else "Looped"
    path = Path(args.data_dir) / folder
    if not path.exists():
        raise FileNotFoundError(
            f"Carpeta de datos no encontrada: {path}\n"
            f"Ajusta --data_dir para que apunte a la raíz del dataset de Mendeley."
        )
    return path


def cache_path(args) -> Path:
    fname = f"{args.transform}_{args.topology}_{args.task}.h5"
    return Path(args.cache_dir) / fname


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline — computes cache if missing, never loads full array into RAM
# ─────────────────────────────────────────────────────────────────────────────

def build_cache_if_needed(args) -> Path:
    """
    Compute time-frequency images and write them to an HDF5 cache file.
    If the cache already exists, this is a no-op.

    Returns the path to the (possibly freshly created) cache file.
    The caller reads from it lazily via HDF5Dataset — no full array is
    ever returned from this function.
    """
    cp = cache_path(args)

    if cp.exists() and not args.no_cache:
        print(f"[data] Caché encontrado → {cp}")
        # Print size info without loading
        with h5py.File(str(cp), "r") as f:
            n_tr = len(f["y_train"])
            n_te = len(f["y_test"])
            shape = f["x_train"].shape[1:]
        print(f"[data] x_train: ({n_tr}, {shape[0]}, {shape[1]}, {shape[2]})  "
              f"x_test: ({n_te}, ...)")
        return cp

    # ── 1. Read CSV signals ───────────────────────────────────────────────
    print("[data] Caché no encontrado — leyendo CSVs...")
    data_path = resolve_data_path(args)

    signals_dict, labels_dict = load_signals_from_csv(
        data_dir=str(data_path),
        task=args.task,
        sample_rate=25600,
        downsample_factor=1,
        fraction_to_include=1.0,
        test_size=0.2,
        seed=args.seed,
    )

    # ── 2. Wavelet denoising ──────────────────────────────────────────────
    print("[data] Aplicando wavelet denoising...")
    signals_dict["training"] = denoise_signal_batch(signals_dict["training"])
    signals_dict["testing"]  = denoise_signal_batch(signals_dict["testing"])

    labels_dict["training"] = np.array(labels_dict["training"], dtype=np.int64)
    labels_dict["testing"]  = np.array(labels_dict["testing"],  dtype=np.int64)

    # ── 3. Time-frequency transform ───────────────────────────────────────
    if args.transform == "cwt":
        print("[data] Calculando escalogramas CWT (puede tardar)...")
        images, labels_dict = calculate_scalograms_with_padding_modes(
            signals_dict, labels_dict, modes=["symmetric"]
        )
    else:
        print("[data] Calculando espectrogramas STFT...")
        images, labels_dict = generate_log_spectrograms(signals_dict, labels_dict)

    # ── 4. Write cache — one sample at a time to avoid RAM spike ─────────
    cp.parent.mkdir(parents=True, exist_ok=True)

    # Determine shape from first sample
    sample0 = images["training"][0]           # (H, W)
    h, w    = sample0.shape
    n_tr    = len(images["training"])
    n_te    = len(images["testing"])

    print(f"[data] Guardando caché → {cp}")
    print(f"       Imágenes: ({h}, {w}) | train={n_tr} | test={n_te}")

    with h5py.File(str(cp), "w") as f:
        ds_xtr = f.create_dataset("x_train", shape=(n_tr, 1, h, w),
                                  dtype="float32", compression="gzip", chunks=(1, 1, h, w))
        ds_ytr = f.create_dataset("y_train", data=labels_dict["training"])

        for i, img in enumerate(images["training"]):
            ds_xtr[i, 0] = img.astype(np.float32)

        ds_xte = f.create_dataset("x_test", shape=(n_te, 1, h, w),
                                  dtype="float32", compression="gzip", chunks=(1, 1, h, w))
        ds_yte = f.create_dataset("y_test", data=labels_dict["testing"])

        for i, img in enumerate(images["testing"]):
            ds_xte[i, 0] = img.astype(np.float32)

    print("[data] Caché guardado. La próxima ejecución lo cargará directo.")
    return cp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    name   = experiment_name(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Experimento : {name}")
    print(f"  Dispositivo : {device}")
    if device.type == "cpu":
        print()
        print("  ⚠️  ADVERTENCIA: No se detectó GPU.")
        print("  En Colab: Runtime → Change runtime type → T4 GPU")
        print("  CWT en CPU puede tardar varias horas por experimento.")
    print("=" * 60)

    set_seed(args.seed)

    # ── Build or locate cache (no full array in RAM) ──────────────────────
    cp = build_cache_if_needed(args)
    is_binary = args.task == "binary"

    # ── Lazy DataLoaders — read H5 on demand ─────────────────────────────
    loaders = make_h5_loaders(
        h5_path=str(cp),
        is_binary=is_binary,
        val_size=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_type = "pretrained" if args.pretrained else "scratch"
    model = build_model(model_type=model_type, transform=args.transform,
                        task=args.task, device=device)

    # ── Training ─────────────────────────────────────────────────────────
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out / "best_model.pt")

    results = train(
        model, loaders["train"], loaders["val"],
        is_binary=is_binary,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_path=ckpt_path,
        device=device,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n── Evaluación en test set ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    class_names = (list(BINARY_LABELS.values())
                   if is_binary else list(MULTICLASS_LABELS.values()))

    y_true, y_pred, y_score = get_predictions(
        model, loaders["test"], device, is_binary
    )
    print_report(y_true, y_pred, class_names=class_names)

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        plot_training_history(results["history"],
                              save_path=str(out / "training_history.png"))
        plot_confusion_matrix(y_true, y_pred, class_names=class_names,
                              title=f"Confusion Matrix — {name}",
                              save_path=str(out / "confusion_matrix.png"))
        plot_roc_curve(y_true, y_score, is_binary=is_binary,
                       class_names=class_names,
                       save_path=str(out / "roc_curve.png"))
        plot_precision_recall(y_true, y_score, is_binary=is_binary,
                              class_names=class_names,
                              save_path=str(out / "pr_curve.png"))
        print(f"[output] Plots guardados en {out}/")

    print(f"\n[output] Checkpoint  : {ckpt_path}")
    print(f"[output] Best val acc: {results['best_val_acc']:.4f}")
    print(f"[output] Tiempo      : {results['elapsed_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
