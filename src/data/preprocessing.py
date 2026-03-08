"""
preprocessing.py
----------------
Wavelet denoising and signal preprocessing utilities for the
water-pipe leak detection pipeline.
"""

import numpy as np
import pywt
from scipy import signal
from tqdm import tqdm


class WaveletDenoising:
    """
    Applies wavelet-based denoising to 1-D acoustic signals.

    Parameters
    ----------
    normalize : bool
        Normalize the signal to [-1, 1] before denoising.
    wavelet : str
        PyWavelets wavelet name (e.g. 'db4').
    level : int
        Decomposition level.
    thr_mode : str
        Thresholding mode: 'soft' or 'hard'.
    method : str
        Threshold estimation method: 'universal' or 'bayes'.
    """

    def __init__(
        self,
        normalize: bool = True,
        wavelet: str = "db4",
        level: int = 3,
        thr_mode: str = "soft",
        method: str = "universal",
    ):
        self.normalize = normalize
        self.wavelet = wavelet
        self.level = level
        self.thr_mode = thr_mode
        self.method = method

    def _universal_threshold(self, coeffs: np.ndarray) -> float:
        sigma = np.median(np.abs(coeffs)) / 0.6745
        return sigma * np.sqrt(2 * np.log(len(coeffs)))

    def _bayes_threshold(self, coeffs: np.ndarray) -> float:
        sigma_n = np.median(np.abs(coeffs)) / 0.6745
        sigma_x = max(np.sqrt(np.mean(coeffs**2) - sigma_n**2), 1e-8)
        return sigma_n**2 / sigma_x

    def denoise(self, sig: np.ndarray) -> np.ndarray:
        if self.normalize:
            max_val = np.max(np.abs(sig))
            if max_val > 0:
                sig = sig / max_val

        coeffs = pywt.wavedec(sig, self.wavelet, level=self.level)
        detail_coeffs = coeffs[1:]

        denoised_detail = []
        for c in detail_coeffs:
            if self.method == "bayes":
                thr = self._bayes_threshold(c)
            else:
                thr = self._universal_threshold(c)
            denoised_detail.append(pywt.threshold(c, thr, mode=self.thr_mode))

        denoised_coeffs = [coeffs[0]] + denoised_detail
        return pywt.waverec(denoised_coeffs, self.wavelet)


def pad_signal(sig: np.ndarray, pad_width, mode: str = "symmetric") -> np.ndarray:
    """
    Pad a 1-D signal using PyWavelets padding modes.

    Parameters
    ----------
    sig : np.ndarray
        Input signal.
    pad_width : int or tuple of (left, right)
        Number of samples to pad on each side.
    mode : str
        Padding mode accepted by ``pywt.pad``.

    Returns
    -------
    np.ndarray
        Padded signal.
    """
    if mode == "none":
        return sig
    return pywt.pad(sig, pad_width, mode)


def load_signals_from_h5(filepath: str):
    """
    Load pre-processed signals and labels from an HDF5 file.

    Returns
    -------
    dict
        Keys: 'x_train', 'x_test', 'y_train', 'y_test'.
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        data = {k: f[k][:] for k in f.keys()}
    return data


def denoise_signal_batch(
    signals: list,
    wavelet: str = "db4",
    level: int = 3,
    thr_mode: str = "soft",
    method: str = "universal",
    normalize: bool = True,
) -> list:
    """
    Apply WaveletDenoising to a list of signals.

    Returns
    -------
    list of np.ndarray
    """
    denoiser = WaveletDenoising(
        normalize=normalize,
        wavelet=wavelet,
        level=level,
        thr_mode=thr_mode,
        method=method,
    )
    return [denoiser.denoise(s) for s in tqdm(signals, desc="Denoising")]
