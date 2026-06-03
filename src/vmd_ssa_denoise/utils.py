"""
Utility functions for signal processing and evaluation metrics.
"""

import numpy as np
from typing import Tuple, Optional


def calculate_snr(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.

    SNR = 10 * log10(P_signal / P_noise)

    Parameters
    ----------
    clean_signal : np.ndarray
        The original clean (noise-free) signal.
    denoised_signal : np.ndarray
        The denoised signal.

    Returns
    -------
    float
        SNR value in dB.
    """
    noise = clean_signal - denoised_signal
    p_signal = np.mean(clean_signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0:
        return np.inf
    return 10.0 * np.log10(p_signal / p_noise)


def calculate_rmse(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error between two signals.

    Parameters
    ----------
    signal1, signal2 : np.ndarray
        Signals to compare.

    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(np.mean((signal1 - signal2) ** 2))


def calculate_mae(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between two signals.

    Parameters
    ----------
    signal1, signal2 : np.ndarray
        Signals to compare.

    Returns
    -------
    float
        MAE value.
    """
    return np.mean(np.abs(signal1 - signal2))


def calculate_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two signals.

    Parameters
    ----------
    signal1, signal2 : np.ndarray
        Signals to compare.

    Returns
    -------
    float
        Correlation coefficient.
    """
    return np.corrcoef(signal1, signal2)[0, 1]


def add_noise(signal: np.ndarray, snr_db: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add white Gaussian noise to a signal at a specified SNR level.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal.
    snr_db : float
        Target SNR in dB.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    p_signal = np.mean(signal ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    p_noise = p_signal / snr_linear
    noise = rng.normal(0, np.sqrt(p_noise), len(signal))
    return signal + noise


def generate_test_signal(
    n_samples: int = 2000,
    fs: float = 1000.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic explosion shockwave-like test signal.

    Creates a signal with a sharp transient (simulating shockwave arrival)
    followed by decaying oscillations, mimicking a typical blast pressure waveform.

    Parameters
    ----------
    n_samples : int, optional
        Number of sample points. Default: 2000.
    fs : float, optional
        Sampling frequency (Hz). Default: 1000.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (time_array, signal_array)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, n_samples / fs, n_samples)

    # Shockwave arrival time
    t_arrival = 0.3

    # Friedlander waveform approximation for blast overpressure
    # P(t) = P_s * (1 - t/t_pos) * exp(-b * t / t_pos)
    P_s = 1.0  # Peak overpressure (normalized)
    t_pos = 0.15  # Positive phase duration
    b = 4.0  # Decay coefficient

    # Friedlander waveform
    dt = t - t_arrival
    mask = dt >= 0
    signal = np.zeros_like(t)
    signal[mask] = (
        P_s
        * (1.0 - dt[mask] / t_pos)
        * np.exp(-b * dt[mask] / t_pos)
    )

    # Add secondary oscillations (reflected wave / after-shock)
    signal += 0.15 * np.sin(2 * np.pi * 15.0 * (t - t_arrival)) * np.exp(-5.0 * dt)
    signal += 0.08 * np.sin(2 * np.pi * 45.0 * (t - t_arrival)) * np.exp(-10.0 * dt)
    signal += 0.05 * np.sin(2 * np.pi * 120.0 * t) * np.exp(-3.0 * dt)

    # Only keep positive time after arrival
    signal[~mask] = 0

    # Normalize
    signal = signal / np.max(np.abs(signal) + 1e-10)

    return t, signal


def generate_test_signal_noisy(
    n_samples: int = 2000,
    fs: float = 1000.0,
    snr_db: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a noisy test signal (clean + noise).

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    fs : float
        Sampling frequency (Hz).
    snr_db : float
        SNR of the noisy signal (dB).
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple
        (time_array, clean_signal, noisy_signal)
    """
    t, clean = generate_test_signal(n_samples, fs, seed)
    noisy = add_noise(clean, snr_db, seed)
    return t, clean, noisy
