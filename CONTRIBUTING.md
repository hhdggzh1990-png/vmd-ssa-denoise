# Contributing to vmd-ssa-denoise

Thank you for your interest in contributing to vmd-ssa-denoise! This document provides guidelines for contributing.

## Getting Started

### Prerequisites

- Python 3.9+
- `pip` or `uv` package manager
- Git

### Setup

```bash
git clone https://github.com/hhdggzh1990-png/vmd-ssa-denoise.git
cd vmd-ssa-denoise
pip install -e ".[dev]"
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** and add tests
4. **Run tests**: `pytest tests/ -v`
5. **Lint code**: `ruff check src/ && ruff format src/`
6. **Commit**: `git commit -m "Description of your changes"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Open a Pull Request**

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to all public functions and classes
- Keep functions focused and under 50 lines when possible

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add new optimization algorithm support
fix: correct VMD convergence check
docs: update README installation instructions
test: add edge case tests for SSA optimizer
refactor: simplify mode selection logic
```

## Reporting Issues

When reporting bugs, please include:

- Python version
- Package version
- Minimal reproducible example
- Expected vs. actual behavior
- Any error tracebacks

## Feature Requests

For feature requests, please describe:

- The motivation/use case
- Proposed API design (if applicable)
- Any relevant references or papers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
