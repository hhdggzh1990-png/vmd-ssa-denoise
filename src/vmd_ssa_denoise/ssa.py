"""
Sparrow Search Algorithm (SSA) optimizer for VMD parameter tuning.

Reference:
    Xue, J., & Shen, B. (2020).
    A novel swarm intelligence optimization approach: sparrow search algorithm.
    Science China Information Sciences, 63(7), 1-15.
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional


class SSAOptimizer:
    """
    Sparrow Search Algorithm for optimizing VMD parameters.

    Optimizes the number of modes (K) and penalty factor (alpha) of VMD
    to minimize a user-defined fitness function (e.g., denoising error, SNR).

    Parameters
    ----------
    pop_size : int, optional
        Population size (number of sparrows). Default: 30.
    max_iter : int, optional
        Maximum number of iterations. Default: 50.
    dim : int, optional
        Dimension of search space (number of parameters). Default: 2
        (K and alpha).
    lb : array-like, optional
        Lower bounds for each dimension. Default: [2, 100].
    ub : array-like, optional
        Upper bounds for each dimension. Default: [10, 5000].
    alarm_threshold : float, optional
        Safety awareness alarm value. Default: 0.8.
    discoverer_ratio : float, optional
        Proportion of discoverers in the population. Default: 0.2.
   警戒者_ratio : float, optional
        Proportion of scouts. Default: 0.1.
    safety_threshold : float, optional
        Safety threshold for position update. Default: 0.2.
    verbose : bool, optional
        Whether to print progress. Default: True.
    """

    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 50,
        dim: int = 2,
        lb=None,
        ub=None,
        alarm_threshold: float = 0.8,
        discoverer_ratio: float = 0.2,
        safety_threshold: float = 0.2,
        verbose: bool = True,
    ):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = dim
        self.lb = np.array(lb if lb is not None else [2.0, 100.0], dtype=np.float64)
        self.ub = np.array(ub if ub is not None else [10.0, 5000.0], dtype=np.float64)
        self.alarm_threshold = alarm_threshold
        self.discoverer_ratio = discoverer_ratio
        self.safety_threshold = safety_threshold
        self.verbose = verbose

        if len(self.lb) != self.dim or len(self.ub) != self.dim:
            raise ValueError("Length of lb and ub must equal dim.")

        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = np.inf
        self.fitness_history: list = []
        self.position_history: list = []

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
    ) -> Tuple[np.ndarray, float]:
        """
        Run the SSA optimization.

        Parameters
        ----------
        fitness_fn : callable
            Fitness function that takes a parameter vector (K_int, alpha)
            and returns a fitness value (lower is better).

        Returns
        -------
        tuple
            (best_position, best_fitness) where best_position is the optimal
            parameter vector and best_fitness is the corresponding fitness.
        """
        # Initialize population
        population = self._initialize_population()
        fitness = np.array([fitness_fn(ind) for ind in population])

        # Record best
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_position = population[best_idx].copy()

        # Number of discoverers and scouts
        n_discoverers = int(self.pop_size * self.discoverer_ratio)

        for iteration in range(self.max_iter):
            # Sort by fitness
            sorted_indices = np.argsort(fitness)

            # Update alarm value (decreases over iterations)
            R2 = np.random.uniform(0, 1, self.pop_size)

            # Update discoverers (producers)
            for i in range(self.pop_size):
                if i < n_discoverers:
                    # Producers search widely
                    if R2[i] < self.alarm_threshold:
                        # Random search with caution
                        population[i] = population[i] * np.exp(
                            -i / (np.random.uniform(0, 1) * self.max_iter)
                        )
                    else:
                        # Random walk
                        population[i] = population[i] + np.random.uniform(-1, 1, self.dim) * np.ones(self.dim)
                else:
                    # Followers follow the best producer
                    if i < self.pop_size / 2:
                        # Positive跟随
                        population[i] = (
                            np.random.uniform(0, 1, self.dim)
                            * np.exp(
                                (population[0] - population[i]) / (i ** 2)
                            )
                        )
                    else:
                        # Negative following
                        population[i] = np.random.uniform(0, 1, self.dim) * (
                            self.best_position - population[i]
                        )

                # Apply boundary constraints
                population[i] = self._clip(population[i])

            # Update scouts (awareness of danger)
            for i in range(self.pop_size):
                # Simulate danger
                danger = np.random.uniform(0, 1, self.dim)
                for d in range(self.dim):
                    if danger[d] < self.safety_threshold:
                        if population[i, d] == self.lb[d]:
                            population[i, d] = self.lb[d] + np.random.uniform(0, 1) * abs(population[i, d] - self.lb[d])
                        elif population[i, d] == self.ub[d]:
                            population[i, d] = self.ub[d] - np.random.uniform(0, 1) * abs(self.ub[d] - population[i, d])
                        else:
                            rand_dir = np.random.uniform(-1, 1)
                            population[i, d] = population[i, d] + rand_dir * np.ones(self.dim)[d] * (self.ub[d] - self.lb[d]) / (self.max_iter * 2)

                population[i] = self._clip(population[i])

            # Evaluate fitness
            fitness = np.array([fitness_fn(ind) for ind in population])

            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_position = population[current_best_idx].copy()

            self.fitness_history.append(self.best_fitness)
            self.position_history.append(self.best_position.copy())

            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"  Iteration {iteration + 1}/{self.max_iter} | "
                    f"Best fitness: {self.best_fitness:.6f} | "
                    f"Best params: K={int(self.best_position[0])}, "
                    f"alpha={self.best_position[1]:.0f}"
                )

        return self.best_position, self.best_fitness

    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        population = np.random.uniform(
            low=self.lb[np.newaxis, :],
            high=self.ub[np.newaxis, :],
            size=(self.pop_size, self.dim),
        )
        # Round K dimension to nearest integer conceptually
        # (kept as float during optimization for continuous search)
        return population

    def _clip(self, position: np.ndarray) -> np.ndarray:
        """Clip position to be within bounds."""
        return np.clip(position, self.lb, self.ub)

    def get_optimal_params(
        self, param_names: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Get optimal parameters as a dictionary.

        Parameters
        ----------
        param_names : list of str, optional
            Names for each parameter. Default: ['K', 'alpha'].

        Returns
        -------
        dict
            Dictionary mapping parameter names to optimal values.
        """
        if self.best_position is None:
            raise RuntimeError("Must call optimize() first.")

        if param_names is None:
            param_names = ["K", "alpha"]

        return {
            name: int(val) if i == 0 else val
            for i, (name, val) in enumerate(zip(param_names, self.best_position))
        }
