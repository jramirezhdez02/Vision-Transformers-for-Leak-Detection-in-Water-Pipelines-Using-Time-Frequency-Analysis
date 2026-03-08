"""
test_pipeline.py
----------------
Prueba el pipeline completo con datos SINTÉTICOS (señales aleatorias).
No necesita el dataset de Mendeley ni GPU.

Corre este script en Colab o localmente ANTES de usar los datos reales
para confirmar que todo funciona sin errores.

Uso
---
    python test_pipeline.py              # prueba rápida (~1 min en CPU)
    python test_pipeline.py --full       # prueba todas las combinaciones posibles
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def header(msg):
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print('─' * 60)

def ok(msg):   print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}"); sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic signal generator
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_signals(n_samples=20, signal_len=25600, n_classes=5, seed=42):
    """
    Generate random acoustic signals and integer class labels.
    signal_len = 25 600 samples @ 25.6 kHz = 1 second of audio.
    """
    rng = np.random.default_rng(seed)
    signals = [rng.standard_normal(signal_len).astype(np.float32) for _ in range(n_samples)]
    labels  = rng.integers(0, n_classes, size=n_samples)
    return signals, labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. Individual unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_imports():
    header("1 · Imports")
    try:
        from src.data.preprocessing import WaveletDenoising, denoise_signal_batch
        from src.data.transforms    import generate_log_spectrograms
        from src.data.dataset       import make_dataloaders, BINARY_LABELS, MULTICLASS_LABELS
        from src.models.vit         import build_model, TRANSFORM_CONFIGS, TASK_CONFIGS
        from src.training.trainer   import set_seed, train
        from src.utils.metrics      import get_predictions, plot_confusion_matrix
        ok("Todos los módulos importan correctamente")
    except ImportError as e:
        fail(f"Import error: {e}")


def test_wavelet_denoising():
    header("2 · Wavelet denoising")
    from src.data.preprocessing import WaveletDenoising, denoise_signal_batch

    sig = np.random.randn(25600).astype(np.float32)
    denoiser = WaveletDenoising(wavelet="db4", level=3)
    out = denoiser.denoise(sig)
    assert out.shape[0] >= len(sig) - 10, "Output length mismatch"
    ok(f"Single signal: in {sig.shape} → out {out.shape}")

    batch_out = denoise_signal_batch([sig] * 4)
    assert len(batch_out) == 4
    ok(f"Batch of 4 signals denoised")


def test_stft_transform():
    header("3 · STFT log-spectrogram")
    from src.data.transforms import generate_log_spectrograms

    signals = [np.random.randn(25600).astype(np.float32) for _ in range(6)]
    labels  = np.array([0, 1, 0, 1, 2, 3])

    specs, lbls = generate_log_spectrograms(
        {"training": signals[:4], "testing": signals[4:]},
        {"training": labels[:4],  "testing": labels[4:]},
    )
    assert "training" in specs and "testing" in specs
    h, w = specs["training"][0].shape
    assert (h, w) == (272, 112), f"Expected (272, 112) after padding, got ({h}, {w})"
    ok(f"Spectrogram shape: ({h} × {w})  ✓  (padded from raw 257×101 to 272×112)")


def test_build_model():
    header("4 · build_model() — todas las combinaciones")
    from src.models.vit import build_model

    device = torch.device("cpu")
    combos = [
        ("scratch",    "cwt",  "binary"),
        ("scratch",    "cwt",  "multiclass"),
        ("scratch",    "stft", "binary"),
        ("scratch",    "stft", "multiclass"),
    ]
    for model_type, transform, task in combos:
        model = build_model(model_type, transform, task, device=device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        ok(f"{model_type:12s} | {transform:4s} | {task:11s} → {n_params:.1f}M params")


def test_build_model_validation():
    header("5 · build_model() — validación de inputs incorrectos")
    from src.models.vit import build_model

    for bad_args, expected_msg in [
        ({"transform": "fourier"}, "transform must be one of"),
        ({"task": "regression"},   "task must be one of"),
        ({"model_type": "xlnet"},  "model_type must be"),
    ]:
        kwargs = {"model_type": "scratch", "transform": "cwt", "task": "binary"}
        kwargs.update(bad_args)
        try:
            build_model(**kwargs)
            fail(f"Debería haber lanzado ValueError con args={bad_args}")
        except ValueError as e:
            ok(f"ValueError correcto para {bad_args}: '{str(e)[:50]}...'")


def test_forward_pass():
    header("6 · Forward pass con tensor sintético")
    from src.models.vit import build_model, TRANSFORM_CONFIGS

    device = torch.device("cpu")

    for transform in ["cwt", "stft"]:
        for task in ["binary", "multiclass"]:
            model = build_model("scratch", transform, task, device=device)
            model.eval()

            h, w = TRANSFORM_CONFIGS[transform]["img_size"]
            x = torch.randn(2, 1, h, w)  # batch of 2

            with torch.no_grad():
                out = model(x)

            expected_dim = 1 if task == "binary" else 5
            assert out.shape == (2, expected_dim), \
                f"Expected (2, {expected_dim}), got {out.shape}"
            ok(f"transform={transform} task={task} → output shape {tuple(out.shape)}")


def test_dataloaders():
    header("7 · make_dataloaders()")
    from src.data.dataset import make_dataloaders

    # Fake STFT-like images: (N, H, W) — channel dim absent, should be added automatically
    x = np.random.randn(60, 272, 112).astype(np.float32)
    y = np.random.randint(0, 2, size=60)

    loaders = make_dataloaders(x, y, is_binary=True, batch_size=8, num_workers=0)
    assert set(loaders.keys()) == {"train", "val"}, \
        f"Expected keys {{'train','val'}}, got {set(loaders.keys())}"

    xb, yb = next(iter(loaders["train"]))
    assert xb.shape[1] == 1, f"Expected channel dim 1, got {xb.shape[1]}"
    ok(f"Train batch: x={tuple(xb.shape)}  y={tuple(yb.shape)}")
    ok("Channel dimension añadida automáticamente ✓")
    ok("make_dataloaders devuelve train + val (test se construye aparte en train.py) ✓")


def test_training_loop():
    header("8 · Training loop — 2 epochs con datos sintéticos")
    from src.models.vit      import build_model, TRANSFORM_CONFIGS
    from src.data.dataset    import make_dataloaders
    from src.training.trainer import set_seed, train

    set_seed(42)
    device = torch.device("cpu")
    transform, task = "stft", "binary"   # STFT es más pequeño → más rápido en CPU

    h, w = TRANSFORM_CONFIGS[transform]["img_size"]  # (272, 112)
    # Pass (N, 1, H, W) — channel dim already present, make_dataloaders handles both
    x = np.random.randn(40, 1, h, w).astype(np.float32)
    y = np.random.randint(0, 2, size=40)

    loaders = make_dataloaders(x, y, is_binary=True, batch_size=8, num_workers=0)
    model   = build_model("scratch", transform, task, device=device)

    t0 = time.time()
    results = train(
        model,
        loaders["train"],
        loaders["val"],
        is_binary=True,
        num_epochs=2,
        patience=5,
        device=device,
    )
    elapsed = time.time() - t0

    assert "history" in results
    assert len(results["history"]["train_loss"]) == 2
    ok(f"2 epochs completados en {elapsed:.1f}s")
    ok(f"Train loss epochs: {[f'{l:.4f}' for l in results['history']['train_loss']]}")


def test_metrics():
    header("9 · Métricas y plots (sin mostrar ventana)")
    import matplotlib
    matplotlib.use("Agg")  # backend sin pantalla

    from src.utils.metrics import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_precision_recall,
        plot_training_history,
        print_report,
    )

    y_true  = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred  = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    y_score = np.array([0.2, 0.6, 0.8, 0.4, 0.3, 0.9, 0.7, 0.1])

    print_report(y_true, y_pred, class_names=["no_leak", "leak"])
    plot_confusion_matrix(y_true, y_pred, class_names=["no_leak", "leak"])
    plot_roc_curve(y_true, y_score, is_binary=True)
    plot_precision_recall(y_true, y_score, is_binary=True)

    history = {
        "train_loss": [0.9, 0.7, 0.5],
        "val_loss":   [1.0, 0.8, 0.6],
        "train_acc":  [0.5, 0.6, 0.7],
        "val_acc":    [0.4, 0.55, 0.65],
    }
    plot_training_history(history)
    ok("Todos los plots generados sin errores")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

FAST_TESTS = [
    test_imports,
    test_wavelet_denoising,
    test_stft_transform,
    test_build_model,
    test_build_model_validation,
    test_forward_pass,
    test_dataloaders,
    test_metrics,
]

FULL_TESTS = FAST_TESTS + [
    test_training_loop,   # tarda ~1 min en CPU
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="Incluir test del training loop (~1 min)")
    args = p.parse_args()

    tests = FULL_TESTS if args.full else FAST_TESTS

    print("\n" + "═" * 60)
    print("  Pipeline test — water pipe leak detection ViT")
    print("═" * 60)

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except SystemExit:
            failed += 1
        except Exception as e:
            print(f"\n  ✗  Excepción inesperada en {test_fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "═" * 60)
    print(f"  Resultado: {passed} pasados  |  {failed} fallidos")
    print("═" * 60 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
