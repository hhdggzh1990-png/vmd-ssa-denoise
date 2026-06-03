"""
VMDDenoiser - Main denoising pipeline combining VMD and SSA optimization.

Provides a high-level interface for adaptive signal denoising using
SSA-optimized VMD parameters.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .vmd import VMD
from .ssa import SSAOptimizer
from .utils import calculate_snr, calculate_rmse


class VMDDenoiser:
    """
    Adaptive denoiser using SSA-optimized VMD.

    Automatically finds optimal VMD parameters (K, alpha) using the
    Sparrow Search Algorithm to maximize denoising performance.

    Parameters
    ----------
    ssa_pop_size : int, optional
        SSA population size. Default: 20.
    ssa_max_iter : int, optional
        SSA maximum iterations. Default: 30.
    k_range : tuple, optional
        Search range for K (number of modes). Default: (2, 8).
    alpha_range : tuple, optional
        Search range for alpha (penalty factor). Default: (100, 3000).
    mode_selection : str, optional
        Strategy for selecting modes for reconstruction.
        'energy' - select modes based on energy contribution (default).
        'correlation' - select modes with highest correlation to signal.
        'all' - use all modes.
    verbose : bool, optional
        Whether to print progress. Default: True.
    """

    def __init__(
        self,
        ssa_pop_size: int = 20,
        ssa_max_iter: int = 30,
        k_range: Tuple[int, int] = (2, 8),
        alpha_range: Tuple[float, float] = (100, 3000),
        mode_selection: str = "energy",
        verbose: bool = True,
    ):
        self.ssa_pop_size = ssa_pop_size
        self.ssa_max_iter = ssa_max_iter
        self.k_range = k_range
        self.alpha_range = alpha_range
        self.mode_selection = mode_selection
        self.verbose = verbose

        self.optimal_k: Optional[int] = None
        self.optimal_alpha: Optional[float] = None
        self.vmd: Optional[VMD] = None
        self.selected_modes: Optional[List[int]] = None

    def fit(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        reference: Optional[np.ndarray] = None,
    ) -> "VMDDenoiser":
        """
        Find optimal VMD parameters using SSA.

        Parameters
        ----------
        signal : np.ndarray
            The noisy input signal.
        fs : float, optional
            Sampling frequency. Default: 1.0.
        reference : np.ndarray, optional
            Clean reference signal for SNR-based fitness evaluation.
            If not provided, uses a reconstruction-based fitness function.

        Returns
        -------
        self
            Fitted denoiser.
        """
        if self.verbose:
            print("Starting SSA optimization for VMD parameters...")

        def fitness_fn(params: np.ndarray) -> float:
            """Fitness function: minimize reconstruction error."""
            K = int(round(params[0]))
            alpha = float(params[1])

            if K < 2:
                K = 2

            try:
                vmd = VMD(K=K, alpha=alpha, tol=1e-6, max_iter=200)
                modes = vmd.decompose(signal, fs=fs)

                if reference is not None:
                    # Use correlation-based fitness with reference
                    reconstructed = self._select_modes(vmd, modes, signal)
                    snr = calculate_snr(reference, reconstructed)
                    return -snr  # Minimize negative SNR
                else:
                    # Auto fitness: balance between reconstruction and smoothness
                    reconstructed = self._select_modes(vmd, modes, signal)
                    residual = signal - reconstructed
                    # Fitness = noise energy (lower is better)
                    noise_energy = np.mean(residual ** 2)
                    # Penalty for too much information loss
                    signal_energy = np.mean(signal ** 2)
                    info_loss = max(0, signal_energy - noise_energy - 0.5 * np.mean(reconstructed ** 2))
                    return noise_energy + 0.1 * info_loss

            except Exception:
                return 1e10

        optimizer = SSAOptimizer(
            pop_size=self.ssa_pop_size,
            max_iter=self.ssa_max_iter,
            dim=2,
            lb=[float(self.k_range[0]), float(self.alpha_range[0])],
            ub=[float(self.k_range[1]), float(self.alpha_range[1])],
            verbose=self.verbose,
        )

        best_pos, best_fit = optimizer.optimize(fitness_fn)

        self.optimal_k = int(round(best_pos[0]))
        self.optimal_alpha = float(best_pos[1])

        if self.verbose:
            print(f"\nOptimal parameters: K={self.optimal_k}, alpha={self.optimal_alpha:.0f}")
            if reference is not None:
                print(f"Best SNR: {-best_fit:.2f} dB")

        return self

    def denoise(self, signal: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """
        Denoise the signal using the optimal VMD parameters.

        Parameters
        ----------
        signal : np.ndarray
            Noisy input signal.
        fs : float, optional
            Sampling frequency. Default: 1.0.

        Returns
        -------
        np.ndarray
            Denoised signal.
        """
        if self.optimal_k is None or self.optimal_alpha is None:
            raise RuntimeError("Must call fit() before denoise().")

        self.vmd = VMD(K=self.optimal_k, alpha=self.optimal_alpha)
        modes = self.vmd.decompose(signal, fs=fs)
        self.selected_modes = self._get_selected_mode_indices(self.vmd, modes, signal)

        if self.verbose:
            print(f"Selected modes for reconstruction: {self.selected_modes}")

        return self.vmd.reconstruct(self.selected_modes)

    def fit_denoise(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convenience method: fit and denoise in one call.

        Parameters
        ----------
        signal : np.ndarray
            Noisy input signal.
        fs : float, optional
            Sampling frequency. Default: 1.0.
        reference : np.ndarray, optional
            Clean reference for optimization.

        Returns
        -------
        np.ndarray
            Denoised signal.
        """
        self.fit(signal, fs=fs, reference=reference)
        return self.denoise(signal, fs=fs)

    def get_decomposition(self) -> Optional[Dict]:
        """
        Get the full VMD decomposition result.

        Returns
        -------
        dict or None
            Dictionary with keys: 'modes', 'frequencies', 'selected_modes'.
        """
        if self.vmd is None or self.vmd.modes is None:
            return None
        return {
            "modes": self.vmd.modes,
            "frequencies": self.vmd.mode_frequencies,
            "selected_modes": self.selected_modes,
            "optimal_k": self.optimal_k,
            "optimal_alpha": self.optimal_alpha,
        }

    def evaluate(
        self, clean: np.ndarray, denoised: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate denoising performance against a clean reference.

        Parameters
        ----------
        clean : np.ndarray
            Clean (noise-free) signal.
        denoised : np.ndarray
            Denoised signal.

        Returns
        -------
        dict
            Evaluation metrics: snr, rmse, mae.
        """
        from .utils import calculate_mae

        return {
            "snr": calculate_snr(clean, denoised),
            "rmse": calculate_rmse(clean, denoised),
            "mae": calculate_mae(clean, denoised),
        }

    def _select_modes(
        self, vmd: VMD, modes: np.ndarray, signal: np.ndarray
    ) -> np.ndarray:
        """Select and reconstruct signal from optimal modes."""
        indices = self._get_selected_mode_indices(vmd, modes, signal)
        return vmd.reconstruct(indices)

    def _get_selected_mode_indices(
        self, vmd: VMD, modes: np.ndarray, signal: np.ndarray
    ) -> List[int]:
        """Determine which modes to keep for reconstruction."""
        K = modes.shape[0]

        if self.mode_selection == "all":
            return list(range(K))

        elif self.mode_selection == "energy":
            # Keep modes with significant energy contribution
            energies = np.array([np.mean(m ** 2) for m in modes])
            total_energy = np.sum(energies)
            if total_energy == 0:
                return list(range(K))

            # Keep modes that contribute more than average energy
            threshold = total_energy / K * 0.5
            selected = [i for i in range(K) if energies[i] > threshold]

            # Always keep at least 1 mode
            if not selected:
                selected = [np.argmax(energies)]

            return selected

        elif self.mode_selection == "correlation":
            # Keep modes with highest correlation to the signal
            from .utils import calculate_correlation

            correlations = [abs(calculate_correlation(m, signal)) for m in modes]
            # Keep modes with correlation above median
            median_corr = np.median(correlations)
            selected = [i for i in range(K) if correlations[i] >= median_corr]

            if not selected:
                selected = [np.argmax(correlations)]

            return selected

        else:
            raise ValueError(f"Unknown mode_selection: {self.mode_selection}")
