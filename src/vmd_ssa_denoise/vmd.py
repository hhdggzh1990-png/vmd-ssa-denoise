"""
Variational Mode Decomposition (VMD) implementation.

Reference:
    Dragomiretskiy, K., & Zosso, D. (2014).
    Variational mode decomposition.
    IEEE Transactions on Signal Processing, 62(3), 531-544.
"""

import numpy as np
from typing import List, Optional


class VMD:
    """
    Variational Mode Decomposition for adaptive signal decomposition.

    Decomposes a 1D signal into K Intrinsic Mode Functions (IMFs) by
    solving a constrained variational optimization problem using
    the Alternate Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    alpha : float, optional
        Bandwidth constraint parameter (penalty factor).
        Higher values produce narrower bandwidth modes. Default: 2000.
    tau : float, optional
        Noise tolerance parameter. 0 = no noise, >0 = noise robustness.
        Default: 0.0.
    K : int, optional
        Number of modes to decompose into. Default: 3.
    DC : bool, optional
        Whether to include a DC (zero-frequency) component. Default: False.
    init : str, optional
        Initialization method: '1perk' (one power spectrum peak per mode)
        or 'allzeros' (all modes start from zero). Default: '1perk'.
    tol : float, optional
        Convergence tolerance. Default: 1e-7.
    max_iter : int, optional
        Maximum number of iterations. Default: 500.
    """

    def __init__(
        self,
        alpha: float = 2000.0,
        tau: float = 0.0,
        K: int = 3,
        DC: bool = False,
        init: str = "1perk",
        tol: float = 1e-7,
        max_iter: int = 500,
    ):
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

        self.modes: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None

    def decompose(self, signal: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """
        Perform VMD decomposition on the input signal.

        Parameters
        ----------
        signal : np.ndarray
            1D input signal to decompose.
        fs : float, optional
            Sampling frequency (Hz), used for frequency scaling. Default: 1.0.

        Returns
        -------
        np.ndarray
            Decomposed modes, shape (K, len(signal)).
        """
        f = signal.copy().flatten().astype(np.float64)
        N = len(f)

        # Frequency axis (normalized)
        freqs = np.fft.fftfreq(N, d=1.0 / N)

        # Initialize center frequencies
        if self.init == "1perk":
            fft_spectrum = np.abs(np.fft.fft(f))[: N // 2]
            peaks = self._find_spectrum_peaks(fft_spectrum, self.K)
            omega = peaks / N
        elif self.init == "allzeros":
            omega = np.zeros(self.K)
        else:
            raise ValueError(f"Unknown init method: {self.init}")

        omega = omega.astype(np.float64)

        # Dual variable (Lagrange multiplier)
        hat_f = np.fft.fft(f)

        # Mode spectra and previous iteration
        u_hat = np.zeros((self.K, N), dtype=np.complex128)
        u_hat_prev = np.zeros_like(u_hat)

        for iteration in range(self.max_iter):
            # Sum of all modes except current
            sum_u_hat = np.sum(u_hat, axis=0)

            # Update each mode
            for k in range(self.K):
                # Wiener filter update for mode k
                sum_others = sum_u_hat - u_hat[k]
                numerator = hat_f - sum_others
                denominator = 1.0 + self.alpha * (freqs - omega[k]) ** 2
                u_hat[k] = numerator / denominator

            # Update center frequencies
            for k in range(self.K):
                half_spectrum = np.abs(u_hat[k, N // 2 :]) ** 2
                freq_half = freqs[N // 2 :]
                numerator = np.sum(freq_half * half_spectrum)
                denominator = np.sum(half_spectrum) + 1e-12
                if not self.DC and k == 0:
                    omega[k] = 0.0
                else:
                    omega[k] = max(0.0, numerator / denominator)

            # Update sum for convergence check
            sum_u_hat = np.sum(u_hat, axis=0)

            # Check convergence
            diff = np.sqrt(
                np.sum(np.abs(u_hat - u_hat_prev) ** 2) / (np.sum(np.abs(u_hat) ** 2) + 1e-12)
            )

            if diff < self.tol:
                break

            u_hat_prev = u_hat.copy()

        # Convert modes back to time domain
        modes = np.zeros((self.K, N))
        for k in range(self.K):
            modes[k] = np.real(np.fft.ifft(u_hat[k]))

        self.modes = modes
        self.omega = omega * fs
        return modes

    def reconstruct(self, mode_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Reconstruct signal from selected modes.

        Parameters
        ----------
        mode_indices : list of int, optional
            Indices of modes to include. If None, uses all modes.

        Returns
        -------
        np.ndarray
            Reconstructed signal.
        """
        if self.modes is None:
            raise RuntimeError("Must call decompose() before reconstruct().")

        if mode_indices is None:
            mode_indices = list(range(self.K))
        return np.sum(self.modes[mode_indices], axis=0)

    @staticmethod
    def _find_spectrum_peaks(spectrum: np.ndarray, n_peaks: int) -> np.ndarray:
        """Find the indices of the n_peaks highest peaks in the spectrum."""
        if len(spectrum) <= n_peaks:
            return np.arange(len(spectrum)).astype(np.float64)
        peaks = np.argsort(spectrum)[-n_peaks:]
        peaks.sort()
        return peaks.astype(np.float64)

    @property
    def mode_frequencies(self) -> Optional[np.ndarray]:
        """Center frequencies of each mode (Hz)."""
        return self.omega
