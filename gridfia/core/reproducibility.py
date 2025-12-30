"""
Reproducibility utilities for deterministic random operations.

This module provides seed management to ensure reproducible results
across runs, including support for parallel processing.
"""

import random
from contextlib import contextmanager
from typing import Optional, Generator
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SeedManager:
    """
    Manage random seeds for reproducibility across the package.

    This class provides global seed management that affects all random
    operations in GridFIA, including parallel processing workers.

    Examples
    --------
    >>> from gridfia.core.reproducibility import SeedManager
    >>>
    >>> # Set global seed
    >>> SeedManager.set_global_seed(42)
    >>>
    >>> # Use temporary seed for specific operation
    >>> with SeedManager.temporary_seed(123):
    ...     result = some_random_operation()
    """

    _global_seed: Optional[int] = None
    _random_state: Optional[np.random.RandomState] = None

    @classmethod
    def set_global_seed(cls, seed: int) -> None:
        """
        Set global seed for all random operations.

        Parameters
        ----------
        seed : int
            Seed value for random number generators.

        Notes
        -----
        This affects:
        - Python's random module
        - NumPy's random module
        - All GridFIA calculations using random operations
        """
        cls._global_seed = seed
        cls._random_state = np.random.RandomState(seed)

        # Set seeds for standard libraries
        random.seed(seed)
        np.random.seed(seed)

        logger.info(f"Global seed set to {seed}")

    @classmethod
    def get_seed(cls) -> Optional[int]:
        """
        Get current global seed.

        Returns
        -------
        int or None
            Current global seed, or None if not set.
        """
        return cls._global_seed

    @classmethod
    def get_random_state(cls) -> Optional[np.random.RandomState]:
        """
        Get the global RandomState object.

        Returns
        -------
        np.random.RandomState or None
            RandomState object for reproducible random operations.
        """
        return cls._random_state

    @classmethod
    def derive_seed(cls, offset: int = 0) -> int:
        """
        Derive a deterministic seed from the global seed.

        Useful for creating unique but reproducible seeds for
        parallel workers or nested operations.

        Parameters
        ----------
        offset : int
            Offset to add to the global seed.

        Returns
        -------
        int
            Derived seed value.

        Raises
        ------
        ValueError
            If no global seed has been set.
        """
        if cls._global_seed is None:
            raise ValueError("No global seed set. Call set_global_seed() first.")
        return cls._global_seed + offset

    @classmethod
    def get_worker_seed(cls, worker_id: int) -> Optional[int]:
        """
        Get a deterministic seed for a parallel worker.

        Parameters
        ----------
        worker_id : int
            Unique identifier for the worker (0, 1, 2, ...).

        Returns
        -------
        int or None
            Seed for the worker, or None if no global seed is set.
        """
        if cls._global_seed is None:
            return None
        # Use a prime multiplier to spread seeds and avoid collisions
        return cls._global_seed + (worker_id * 997)

    @classmethod
    @contextmanager
    def temporary_seed(cls, seed: int) -> Generator[None, None, None]:
        """
        Context manager for temporary seed override.

        Parameters
        ----------
        seed : int
            Temporary seed to use within the context.

        Yields
        ------
        None

        Examples
        --------
        >>> with SeedManager.temporary_seed(999):
        ...     # Operations here use seed 999
        ...     result = np.random.rand()
        >>> # Original seed is restored here
        """
        # Save current state
        old_random_state = random.getstate()
        old_numpy_state = np.random.get_state()
        old_global_seed = cls._global_seed

        try:
            # Set temporary seed
            random.seed(seed)
            np.random.seed(seed)
            cls._global_seed = seed
            yield
        finally:
            # Restore original state
            random.setstate(old_random_state)
            np.random.set_state(old_numpy_state)
            cls._global_seed = old_global_seed

    @classmethod
    def reset(cls) -> None:
        """
        Reset seed manager to initial state (no global seed).
        """
        cls._global_seed = None
        cls._random_state = None
        logger.info("Seed manager reset")


def set_seed(seed: int) -> None:
    """
    Convenience function to set global seed.

    Parameters
    ----------
    seed : int
        Seed value for random number generators.

    Examples
    --------
    >>> from gridfia.core.reproducibility import set_seed
    >>> set_seed(42)
    """
    SeedManager.set_global_seed(seed)


def get_seed() -> Optional[int]:
    """
    Convenience function to get current global seed.

    Returns
    -------
    int or None
        Current global seed.
    """
    return SeedManager.get_seed()
