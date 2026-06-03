"""
vmd_ssa_denoise - Adaptive denoising of noisy signals using VMD optimized by SSA.

This library implements Variational Mode Decomposition (VMD) combined with
Sparrow Search Algorithm (SSA) for intelligent parameter optimization,
specifically targeting noisy explosion shockwave signal denoising.
"""

__version__ = "0.1.0"
__author__ = "hhdggzh1990-png"

from .vmd import VMD
from .ssa import SSAOptimizer
from .denoise import VMDDenoiser
from .utils import (
    calculate_snr,
    calculate_rmse,
    calculate_mae,
    add_noise,
    generate_test_signal,
    generate_test_signal_noisy,
)

__all__ = [
    "VMD",
    "SSAOptimizer",
    "VMDDenoiser",
    "calculate_snr",
    "calculate_rmse",
    "calculate_mae",
    "add_noise",
    "generate_test_signal",
    "generate_test_signal_noisy",
]
