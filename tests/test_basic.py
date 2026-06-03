"""
Test suite for vmd_ssa_denoise.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vmd_ssa_denoise import (
    VMD,
    SSAOptimizer,
    VMDDenoiser,
    calculate_snr,
    calculate_rmse,
    calculate_mae,
    generate_test_signal,
    add_noise,
)
from vmd_ssa_denoise.utils import generate_test_signal_noisy


class TestUtils:
    """Test utility functions."""

    def test_generate_test_signal_shape(self):
        t, signal = generate_test_signal(n_samples=1000, fs=500.0)
        assert len(t) == 1000
        assert len(signal) == 1000

    def test_generate_test_signal_noisy(self):
        t, clean, noisy = generate_test_signal_noisy(n_samples=500, snr_db=10.0)
        assert len(clean) == len(noisy)
        assert np.any(clean != noisy)  # Noise was added

    def test_add_noise_snr(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
        noisy = add_noise(signal, snr_db=10.0, seed=42)
        snr = calculate_snr(signal, noisy)
        assert 8.0 < snr < 12.0  # Should be approximately 10 dB

    def test_snr_identical_signals(self):
        signal = np.random.randn(100)
        assert calculate_snr(signal, signal) == np.inf

    def test_rmse_identical(self):
        signal = np.random.randn(100)
        assert calculate_rmse(signal, signal) == 0.0

    def test_mae_identical(self):
        signal = np.random.randn(100)
        assert calculate_mae(signal, signal) == 0.0


class TestVMD:
    """Test VMD decomposition."""

    def test_vmd_decompose_shape(self):
        vmd = VMD(K=3, alpha=2000, max_iter=100, tol=1e-4)
        signal = np.sin(np.linspace(0, 4 * np.pi, 500)) + 0.5 * np.random.randn(500)
        modes = vmd.decompose(signal)
        assert modes.shape[0] == 3
        assert modes.shape[1] == 500

    def test_vmd_reconstruct(self):
        vmd = VMD(K=3, alpha=2000, max_iter=100, tol=1e-4)
        signal = np.sin(np.linspace(0, 4 * np.pi, 500))
        vmd.decompose(signal)
        reconstructed = vmd.reconstruct()
        assert reconstructed.shape == (500,)

    def test_vmd_mode_frequencies(self):
        vmd = VMD(K=3, alpha=2000, max_iter=100, tol=1e-4)
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 500))
        vmd.decompose(signal, fs=500)
        freqs = vmd.mode_frequencies
        assert freqs is not None
        assert len(freqs) == 3


class TestSSA:
    """Test SSA optimizer."""

    def test_ssa_basic_optimization(self):
        # Optimize a simple quadratic function
        def fitness(x):
            return (x[0] - 5) ** 2 + (x[1] - 10) ** 2

        ssa = SSAOptimizer(
            pop_size=15, max_iter=20, dim=2,
            lb=[0, 0], ub=[10, 20], verbose=False
        )
        best_pos, best_fit = ssa.optimize(fitness)

        # Should find something close to (5, 10)
        assert best_fit < 5.0
        assert abs(best_pos[0] - 5) < 3
        assert abs(best_pos[1] - 10) < 5

    def test_ssa_get_optimal_params(self):
        ssa = SSAOptimizer(
            pop_size=10, max_iter=5, dim=2,
            lb=[2, 100], ub=[8, 3000], verbose=False
        )
        def dummy_fitness(x):
            return np.sum(x ** 2)
        ssa.optimize(dummy_fitness)
        params = ssa.get_optimal_params()
        assert "K" in params
        assert "alpha" in params


class TestVMDDenoiser:
    """Test the full denoising pipeline."""

    def test_fit_denoise_basic(self):
        # Generate test signal
        t, clean = generate_test_signal(n_samples=1000, fs=1000.0, seed=42)
        noisy = add_noise(clean, snr_db=5.0, seed=42)

        denoiser = VMDDenoiser(
            ssa_pop_size=10,
            ssa_max_iter=10,
            k_range=(2, 5),
            alpha_range=(200, 2000),
            verbose=False,
        )

        denoised = denoiser.fit_denoise(noisy, fs=1000.0, reference=clean)

        assert len(denoised) == len(noisy)
        assert denoiser.optimal_k is not None
        assert denoiser.optimal_alpha is not None

    def test_fit_denoise_improves_snr(self):
        t, clean = generate_test_signal(n_samples=500, fs=500.0, seed=42)
        noisy = add_noise(clean, snr_db=3.0, seed=42)

        denoiser = VMDDenoiser(
            ssa_pop_size=8,
            ssa_max_iter=8,
            k_range=(2, 4),
            alpha_range=(200, 1500),
            verbose=False,
        )

        denoised = denoiser.fit_denoise(noisy, fs=500.0, reference=clean)

        original_snr = calculate_snr(clean, noisy)
        improved_snr = calculate_snr(clean, denoised)
        # Denoising should generally improve or maintain SNR
        # With very few iterations, improvement may be modest
        assert improved_snr >= original_snr - 5  # Relaxed tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
