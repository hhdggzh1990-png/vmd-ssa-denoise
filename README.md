# vmd-ssa-denoise

[![CI](https://github.com/hhdggzh1990-png/vmd-ssa-denoise/actions/workflows/ci.yml/badge.svg)](https://github.com/hhdggzh1990-png/vmd-ssa-denoise/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/PyPI-0.1.0-orange.svg)](https://pypi.org/project/vmd-ssa-denoise/)

**Adaptive Signal Denoising using VMD Optimized by Sparrow Search Algorithm**

A Python library for intelligent denoising of noisy signals — particularly explosion shockwave signals — using **Variational Mode Decomposition (VMD)** with parameters automatically optimized by the **Sparrow Search Algorithm (SSA)**.

## Features

- **VMD Implementation**: Full Variational Mode Decomposition using ADMM optimization
- **SSA Optimizer**: Sparrow Search Algorithm for automatic VMD parameter (K, alpha) optimization
- **Adaptive Mode Selection**: Energy-based and correlation-based strategies for selecting relevant modes
- **Evaluation Metrics**: SNR, RMSE, MAE, and correlation coefficient
- **Synthetic Signal Generator**: Built-in explosion shockwave signal simulation (Friedlander waveform)

## Installation

```bash
pip install vmd-ssa-denoise
```

Or install from source:

```bash
git clone https://github.com/hhdggzh1990-png/vmd-ssa-denoise.git
cd vmd-ssa-denoise
pip install -e .
```

## Quick Start

```python
from vmd_ssa_denoise import VMDDenoiser, generate_test_signal, add_noise

# Generate a synthetic explosion shockwave signal
t, clean_signal = generate_test_signal(n_samples=2000, fs=1000.0)

# Add noise
noisy_signal = add_noise(clean_signal, snr_db=5.0)

# Denoise with automatic parameter optimization
denoiser = VMDDenoiser(
    ssa_pop_size=20,        # Population size
    ssa_max_iter=30,         # Max iterations
    k_range=(2, 8),          # Number of modes search range
    alpha_range=(100, 3000), # Penalty factor search range
    mode_selection="energy" # Mode selection strategy
)

# Option 1: With clean reference (best results)
denoiser.fit(noisy_signal, fs=1000.0, reference=clean_signal)
denoised = denoiser.denoise(noisy_signal, fs=1000.0)

# Option 2: Without reference (auto mode)
denoised = denoiser.fit_denoise(noisy_signal, fs=1000.0)

# Evaluate
results = denoiser.evaluate(clean_signal, denoised)
print(f"SNR improvement: {results['snr']:.2f} dB")
print(f"Optimal K={denoiser.optimal_k}, alpha={denoiser.optimal_alpha:.0f}")
```

## How It Works

### Pipeline Overview

```
Noisy Signal
    |
    v
[SSA Optimizer] -- iteratively tests (K, alpha) combinations
    |
    v
[VMD Decomposition] -- decomposes signal into K modes using optimal params
    |
    v
[Mode Selection] -- keeps signal-relevant modes, discards noise
    |
    v
[Reconstruction] -- sum selected modes → denoised signal
```

### VMD (Variational Mode Decomposition)

VMD decomposes a signal into K narrow-band modes by solving a constrained variational problem:

$$\min_{\{u_k\}, \{\omega_k\}} \left\{ \sum_k \left\| \partial_t \left[ \left( \delta(t) + \frac{j}{\pi t} \right) * u_k(t) \right] e^{-j\omega_k t} \right\|_2^2 \right\}$$

Reference: Dragomiretskiy, K., & Zosso, D. (2014). *IEEE Transactions on Signal Processing*, 62(3), 531-544.

### SSA (Sparrow Search Algorithm)

SSA is a bio-inspired metaheuristic that mimics the foraging behavior of sparrows:
- **Discoverers** (producers): Search for food globally
- **Followers**: Follow the best discoverer
- **Scouts**: Monitor danger and perform random walks

SSA optimizes VMD's K (number of modes) and alpha (bandwidth penalty) to maximize denoising quality.

Reference: Xue, J., & Shen, B. (2020). *Science China Information Sciences*, 63(7), 1-15.

### Application: Explosion Shockwave Denoising

This library is specifically designed for denoising explosion shockwave signals, which are characterized by:
- Sharp transient arrival (shock front)
- Rapid exponential decay (Friedlander waveform)
- Low SNR environments
- Mixed frequency content

The synthetic signal generator produces realistic Friedlander-type blast pressure waveforms for testing.

## API Reference

### VMDDenoiser

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ssa_pop_size` | 20 | SSA population size |
| `ssa_max_iter` | 30 | Maximum SSA iterations |
| `k_range` | (2, 8) | Search range for number of modes |
| `alpha_range` | (100, 3000) | Search range for penalty factor |
| `mode_selection` | "energy" | Mode selection: "energy", "correlation", or "all" |

| Method | Description |
|--------|-------------|
| `fit(signal, fs, reference)` | Find optimal VMD parameters |
| `denoise(signal, fs)` | Apply VMD denoising with optimal params |
| `fit_denoise(signal, fs, reference)` | Fit and denoise in one step |
| `evaluate(clean, denoised)` | Compute SNR, RMSE, MAE |

### VMD

Direct access to VMD decomposition:

```python
from vmd_ssa_denoise import VMD

vmd = VMD(K=4, alpha=2000)
modes = vmd.decompose(signal, fs=1000)
reconstructed = vmd.reconstruct([0, 2])  # Use only modes 0 and 2
```

### SSAOptimizer

Direct access to SSA for custom optimization:

```python
from vmd_ssa_denoise import SSAOptimizer

def my_fitness(params):
    K = int(round(params[0]))
    alpha = float(params[1])
    # Your custom fitness evaluation
    return error_value

ssa = SSAOptimizer(pop_size=30, max_iter=50)
best_pos, best_fit = ssa.optimize(my_fitness)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{vmd_ssa_denoise,
  author = {hhdggzh1990-png},
  title = {vmd-ssa-denoise: Adaptive Signal Denoising using VMD Optimized by SSA},
  year = {2026},
  url = {https://github.com/hhdggzh1990-png/vmd-ssa-denoise}
}
```
