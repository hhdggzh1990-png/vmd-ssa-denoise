# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-03

### Added
- VMD (Variational Mode Decomposition) implementation using ADMM optimization
- SSA (Sparrow Search Algorithm) optimizer for automatic VMD parameter tuning
- High-level VMDDenoiser pipeline with energy-based mode selection
- Synthetic Friedlander explosion shockwave signal generator
- Evaluation metrics: SNR, RMSE, MAE, correlation coefficient
- Utility functions: noise addition, signal generation
- 13 unit tests covering all core modules
- Example usage scripts
- CI/CD pipeline with GitHub Actions (test across Python 3.9-3.13)
- PyPI publish workflow
- Contribution guidelines and code of conduct
- Issue and PR templates

### Documentation
- Comprehensive README with mathematical formulas and usage examples
- Inline docstrings for all public APIs
- Application guide for OpenAI Codex for Open Source program
