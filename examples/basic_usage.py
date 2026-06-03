"""
Basic usage example for vmd_ssa_denoise.

Demonstrates how to denoise a synthetic explosion shockwave signal
using SSA-optimized VMD.
"""

import numpy as np
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vmd_ssa_denoise import VMDDenoiser, generate_test_signal, add_noise, calculate_snr


def main():
    # 1. Generate a synthetic clean signal (explosion shockwave)
    t, clean_signal = generate_test_signal(n_samples=2000, fs=1000.0, seed=42)

    # 2. Add noise (SNR = 5 dB)
    noisy_signal = add_noise(clean_signal, snr_db=5.0, seed=42)

    print(f"Signal length: {len(clean_signal)}")
    print(f"Sampling rate: 1000 Hz")
    print(f"Noise level: SNR = 5 dB")

    # 3. Create denoiser and fit on noisy signal
    denoiser = VMDDenoiser(
        ssa_pop_size=15,       # Number of sparrows (reduce for speed)
        ssa_max_iter=20,        # Max iterations (reduce for speed)
        k_range=(2, 6),         # Search K from 2 to 6
        alpha_range=(200, 2500), # Search alpha from 200 to 2500
        mode_selection="energy",
        verbose=True,
    )

    # Fit: find optimal K and alpha
    denoiser.fit(noisy_signal, fs=1000.0, reference=clean_signal)

    # 4. Denoise
    denoised_signal = denoiser.denoise(noisy_signal, fs=1000.0)

    # 5. Evaluate
    original_snr = calculate_snr(clean_signal, noisy_signal)
    denoised_snr = calculate_snr(clean_signal, denoised_signal)
    metrics = denoiser.evaluate(clean_signal, denoised_signal)

    print("\n" + "=" * 50)
    print("DENOISING RESULTS")
    print("=" * 50)
    print(f"Optimal K:          {denoiser.optimal_k}")
    print(f"Optimal alpha:      {denoiser.optimal_alpha:.0f}")
    print(f"Selected modes:     {denoiser.selected_modes}")
    print(f"Original SNR:       {original_snr:.2f} dB")
    print(f"Denoised SNR:       {denoised_snr:.2f} dB")
    print(f"SNR Improvement:     {denoised_snr - original_snr:.2f} dB")
    print(f"RMSE:               {metrics['rmse']:.6f}")
    print(f"MAE:                {metrics['mae']:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
