"""
metrics.py
----------
Evaluation metrics and visualisation helpers.
Covers binary and multiclass classification tasks.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_binary: bool,
):
    """
    Collect predictions and ground-truth labels from a DataLoader.

    Returns
    -------
    y_true : np.ndarray  (N,)
    y_pred : np.ndarray  (N,)    class predictions
    y_score : np.ndarray         raw probabilities / logits for ROC/PR curves
    """
    model.eval()
    all_true, all_pred, all_score = [], [], []

    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)

        if is_binary:
            prob = torch.sigmoid(out).squeeze(1).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            all_score.append(prob)
        else:
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            all_score.append(prob)

        all_pred.append(pred)
        # yb can be (N,1) for binary or (N,) for multiclass — normalise to (N,)
        all_true.append(yb.view(-1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_score = np.concatenate(all_score)

    return y_true, y_pred, y_score


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def print_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None):
    """Print a full sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    for ax, data, fmt, ttl in zip(
        axes, [cm, cm_norm], ["d", ".2f"], [title, title + " (Normalised)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(ttl)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_binary: bool,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(7, 6))
    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        n_cls = y_score.shape[1]
        classes = list(range(n_cls))
        y_bin = label_binarize(y_true, classes=classes)
        for i in range(n_cls):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            lbl = class_names[i] if class_names else f"Class {i}"
            plt.plot(fpr, tpr, lw=1.5, label=f"{lbl} (AUC={auc(fpr, tpr):.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_binary: bool,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(7, 6))
    if is_binary:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        n_cls = y_score.shape[1]
        classes = list(range(n_cls))
        y_bin = label_binarize(y_true, classes=classes)
        for i in range(n_cls):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            lbl = class_names[i] if class_names else f"Class {i}"
            plt.plot(rec, prec, lw=1.5, label=f"{lbl} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
