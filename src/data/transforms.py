"""
transforms.py
-------------
Time-frequency representation utilities:
  - CWT scalograms   (via fCWT / PyWavelets)
  - STFT log-power spectrograms

Output shapes (after resize/pad to ViT-compatible sizes):
  - CWT  : (50, 2048)  — 50 frequency bins, time cropped to 2048
  - STFT : (272, 112)  — padded from raw (257, 101) to be divisible by patch_size=8
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import numpy as np
import pywt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CWT Scalograms
# ---------------------------------------------------------------------------

def calculate_scalograms_with_padding_modes(
    signals_dict: Dict[str, List[np.ndarray]],
    labels_dict:  Dict[str, List],
    modes: List[str],
    fs: int = 25600,
    library: str = "fcwt",
    target_time_points: int = 2048,
) -> Tuple[Dict, Dict]:
    """
    Compute CWT scalograms and crop the time axis to ``target_time_points``.

    Raw CWT output is (50 freqs × 25600 time).  The time axis is cropped to
    ``target_time_points`` (default 2048) to keep memory manageable and produce
    a shape that is evenly divisible by the ViT patch size (5 × 16).

    Parameters
    ----------
    signals_dict : ``{'training': [...], 'testing': [...]}``
    labels_dict  : corresponding labels
    modes : padding modes to try in order (e.g. ``['symmetric']``)
    fs : sampling frequency (Hz)
    library : ``'fcwt'`` or ``'pywt'``
    target_time_points : crop time axis to this length

    Returns
    -------
    scalograms_dict, labels_dict
        scalograms shape per sample: (50, target_time_points)
    """
    if library == "fcwt":
        return _scalograms_fcwt(signals_dict, labels_dict, modes, fs, target_time_points)
    return _scalograms_pywt(signals_dict, labels_dict, modes, fs, target_time_points)


def _scalograms_fcwt(signals_dict, labels_dict, modes, fs, target_time_points):
    import fcwt

    f0, f1, fn = 100, 12800, 50
    results: Dict = {}

    for split, signals in signals_dict.items():
        scalograms = []
        for sig in tqdm(signals, desc=f"CWT [{split}]"):
            coefs = None
            for mode in modes:
                try:
                    pad = len(sig) // 4
                    padded = pywt.pad(sig, pad, mode) if mode != "none" else sig
                    _, coefs = fcwt.cwt(padded, fs, f0, f1, fn)
                    # Remove padding
                    coefs = coefs[:, pad: pad + len(sig)]
                    break
                except Exception:
                    continue

            if coefs is None:
                # fallback: pywt
                scales = np.arange(1, fn + 1)
                coefs, _ = pywt.cwt(sig, scales, "morl", sampling_period=1 / fs)

            # Crop time axis to target_time_points
            t = coefs.shape[1]
            if t >= target_time_points:
                coefs = coefs[:, :target_time_points]
            else:
                # pad with zeros if signal is shorter (shouldn't happen at 25.6kHz)
                coefs = np.pad(coefs, ((0, 0), (0, target_time_points - t)))

            scalograms.append(np.abs(coefs).astype(np.float32))

        results[split] = scalograms
        gc.collect()

    return results, labels_dict


def _scalograms_pywt(signals_dict, labels_dict, modes, fs, target_time_points):
    scales = np.arange(1, 51)  # 50 voices to match fcwt default

    results: Dict = {}
    for split, signals in signals_dict.items():
        scalograms = []
        for sig in tqdm(signals, desc=f"CWT (pywt) [{split}]"):
            coefs, _ = pywt.cwt(sig, scales, "morl", sampling_period=1 / fs)
            t = coefs.shape[1]
            if t >= target_time_points:
                coefs = coefs[:, :target_time_points]
            else:
                coefs = np.pad(coefs, ((0, 0), (0, target_time_points - t)))
            scalograms.append(np.abs(coefs).astype(np.float32))

        results[split] = scalograms
        gc.collect()

    return results, labels_dict


# ---------------------------------------------------------------------------
# STFT Log-Power Spectrograms
# ---------------------------------------------------------------------------

# Raw STFT output for a 1-second signal at 25600 Hz → (257, 101)
# Pad to (272, 112) so both dims are divisible by patch_size=8
_STFT_TARGET_H = 272
_STFT_TARGET_W = 112


def generate_log_spectrograms(
    signals_dict: Dict[str, List[np.ndarray]],
    labels_dict:  Dict[str, List],
    fs: int = 25600,
) -> Tuple[Dict, Dict]:
    """
    Generate log-power STFT spectrograms, padded to (272 × 112).

    STFT parameters (matching original notebooks, tuned for 25.6 kHz):
      - Frame length : 20 ms → 512 samples
      - Frame shift  : 10 ms → 256 samples
      - FFT size     : 512

    Raw output is (257, 101). Zero-padded to (272, 112) so it is evenly
    divisible by the ViT patch size of 8×8.

    Returns
    -------
    spectrograms_dict, labels_dict
        spectrogram shape per sample: (272, 112)
    """
    from scipy import signal as sp_signal

    nperseg  = int(0.020 * fs)   # 512
    noverlap = int(0.010 * fs)   # 256
    nfft     = nperseg            # 512

    spectrograms_dict: Dict = {}

    for split, signals in signals_dict.items():
        specs = []
        for sig in tqdm(signals, desc=f"STFT [{split}]"):
            _, _, Zxx = sp_signal.stft(
                sig, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft
            )
            log_spec = np.log1p(np.abs(Zxx)).astype(np.float32)  # (257, 101)

            # Zero-pad to (272, 112)
            h, w = log_spec.shape
            padded = np.zeros((_STFT_TARGET_H, _STFT_TARGET_W), dtype=np.float32)
            padded[:h, :w] = log_spec

            specs.append(padded)

        spectrograms_dict[split] = specs
        gc.collect()

    return spectrograms_dict, labels_dict
