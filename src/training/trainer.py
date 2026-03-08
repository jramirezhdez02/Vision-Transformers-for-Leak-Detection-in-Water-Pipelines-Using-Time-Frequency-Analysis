"""
trainer.py
----------
Training loop, validation, and full evaluation for the ViT leak detector.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in tqdm(loader, leave=False, desc="Train"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)

        if is_binary:
            preds = (torch.sigmoid(out) > 0.5).long().squeeze(1)
            correct += (preds == yb.long().squeeze(1)).sum().item()
        else:
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool,
) -> Tuple[float, float]:
    """Evaluate on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)

        if is_binary:
            preds = (torch.sigmoid(out) > 0.5).long().squeeze(1)
            correct += (preds == yb.long().squeeze(1)).sum().item()
        else:
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    is_binary: bool,
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Full training loop with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    is_binary : bool
    num_epochs : int
    lr : float
        Initial learning rate.
    weight_decay : float
    patience : int
        Early-stopping patience (epochs without val improvement).
    save_path : str, optional
        Path to save the best checkpoint (.pt file).
    device : torch.device

    Returns
    -------
    dict with keys ``history``, ``best_val_acc``, ``elapsed_sec``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = (
        nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, is_binary)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, is_binary)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
            f"Val   loss: {va_loss:.4f}  acc: {va_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Checkpoint saved → {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        gc.collect()

    elapsed = time.time() - t0
    best_val_acc = max(history["val_acc"])
    print(f"\nTraining finished in {elapsed / 60:.1f} min | Best val acc: {best_val_acc:.4f}")

    return {"history": history, "best_val_acc": best_val_acc, "elapsed_sec": elapsed}
