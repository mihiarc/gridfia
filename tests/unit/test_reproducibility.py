"""
Unit tests for reproducibility utilities.

Tests seed management, deterministic random operations, and reproducibility
across parallel processing and statistical analysis.
"""

import pytest
import numpy as np
import random
from unittest.mock import patch, MagicMock

from gridfia.core.reproducibility import (
    SeedManager,
    set_seed,
    get_seed,
)


class TestSeedManager:
    """Test suite for SeedManager class."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_initial_state(self):
        """Test that SeedManager starts with no seed set."""
        assert SeedManager.get_seed() is None
        assert SeedManager.get_random_state() is None

    def test_set_global_seed(self):
        """Test setting global seed."""
        SeedManager.set_global_seed(42)

        assert SeedManager.get_seed() == 42
        assert SeedManager.get_random_state() is not None
        assert isinstance(SeedManager.get_random_state(), np.random.RandomState)

    def test_seed_affects_numpy_random(self):
        """Test that setting seed affects numpy random operations."""
        SeedManager.set_global_seed(42)
        result1 = np.random.rand(5)

        SeedManager.set_global_seed(42)
        result2 = np.random.rand(5)

        np.testing.assert_array_equal(result1, result2)

    def test_seed_affects_python_random(self):
        """Test that setting seed affects Python random operations."""
        SeedManager.set_global_seed(42)
        result1 = [random.random() for _ in range(5)]

        SeedManager.set_global_seed(42)
        result2 = [random.random() for _ in range(5)]

        assert result1 == result2

    def test_derive_seed(self):
        """Test deriving seeds from global seed."""
        SeedManager.set_global_seed(100)

        assert SeedManager.derive_seed() == 100
        assert SeedManager.derive_seed(0) == 100
        assert SeedManager.derive_seed(1) == 101
        assert SeedManager.derive_seed(10) == 110

    def test_derive_seed_without_global(self):
        """Test that derive_seed raises error without global seed."""
        with pytest.raises(ValueError, match="No global seed set"):
            SeedManager.derive_seed()

    def test_get_worker_seed(self):
        """Test getting worker-specific seeds."""
        SeedManager.set_global_seed(100)

        # Worker seeds should be deterministic
        seed0 = SeedManager.get_worker_seed(0)
        seed1 = SeedManager.get_worker_seed(1)
        seed2 = SeedManager.get_worker_seed(2)

        assert seed0 == 100  # base seed + 0 * 997
        assert seed1 == 100 + 997  # base seed + 1 * 997
        assert seed2 == 100 + 2 * 997  # base seed + 2 * 997

        # Seeds should be different for different workers
        assert seed0 != seed1
        assert seed1 != seed2

    def test_get_worker_seed_without_global(self):
        """Test that get_worker_seed returns None without global seed."""
        assert SeedManager.get_worker_seed(0) is None
        assert SeedManager.get_worker_seed(5) is None

    def test_temporary_seed_context(self):
        """Test temporary seed context manager."""
        SeedManager.set_global_seed(42)

        # Generate random inside temporary context
        with SeedManager.temporary_seed(999):
            assert SeedManager.get_seed() == 999
            temp_result = np.random.rand(3)

        # After context, original seed should be restored
        assert SeedManager.get_seed() == 42

        # Verify temporary results are reproducible
        with SeedManager.temporary_seed(999):
            temp_result2 = np.random.rand(3)

        np.testing.assert_array_equal(temp_result, temp_result2)

    def test_temporary_seed_restores_random_state(self):
        """Test that temporary seed properly restores random state."""
        SeedManager.set_global_seed(42)

        # Advance random state
        np.random.rand(5)

        # Save expected next values
        SeedManager.set_global_seed(42)
        np.random.rand(5)
        expected_after = np.random.rand(3)

        # Reset and use temporary seed
        SeedManager.set_global_seed(42)
        np.random.rand(5)

        with SeedManager.temporary_seed(999):
            np.random.rand(100)  # Generate many values

        # State should be restored
        actual_after = np.random.rand(3)
        np.testing.assert_array_equal(expected_after, actual_after)

    def test_reset(self):
        """Test resetting SeedManager."""
        SeedManager.set_global_seed(42)
        assert SeedManager.get_seed() == 42

        SeedManager.reset()
        assert SeedManager.get_seed() is None
        assert SeedManager.get_random_state() is None


class TestConvenienceFunctions:
    """Test convenience functions for seed management."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_set_seed_function(self):
        """Test set_seed convenience function."""
        set_seed(123)
        assert get_seed() == 123
        assert SeedManager.get_seed() == 123

    def test_get_seed_function(self):
        """Test get_seed convenience function."""
        assert get_seed() is None

        set_seed(456)
        assert get_seed() == 456


class TestReproducibilityIntegration:
    """Integration tests for reproducibility across modules."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_reproducible_numpy_operations(self):
        """Test that numpy operations are reproducible with seed."""
        def run_calculations():
            return {
                'uniform': np.random.rand(10),
                'normal': np.random.normal(0, 1, 10),
                'choice': np.random.choice([1, 2, 3, 4, 5], size=10),
                'permutation': np.random.permutation(10),
            }

        SeedManager.set_global_seed(42)
        results1 = run_calculations()

        SeedManager.set_global_seed(42)
        results2 = run_calculations()

        np.testing.assert_array_equal(results1['uniform'], results2['uniform'])
        np.testing.assert_array_equal(results1['normal'], results2['normal'])
        np.testing.assert_array_equal(results1['choice'], results2['choice'])
        np.testing.assert_array_equal(results1['permutation'], results2['permutation'])

    def test_reproducible_bootstrap_sampling(self):
        """Test that bootstrap sampling is reproducible."""
        from sklearn.utils import resample

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        SeedManager.set_global_seed(42)
        samples1 = [resample(data, n_samples=5, random_state=42 + i) for i in range(5)]

        SeedManager.set_global_seed(42)
        samples2 = [resample(data, n_samples=5, random_state=42 + i) for i in range(5)]

        for s1, s2 in zip(samples1, samples2):
            np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        SeedManager.set_global_seed(42)
        result1 = np.random.rand(10)

        SeedManager.set_global_seed(123)
        result2 = np.random.rand(10)

        assert not np.allclose(result1, result2)


class TestWorkerSeedReproducibility:
    """Tests for worker seed reproducibility in parallel processing."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_worker_seeds_are_deterministic(self):
        """Test that worker seeds are deterministic given global seed."""
        SeedManager.set_global_seed(42)

        seeds_run1 = [SeedManager.get_worker_seed(i) for i in range(10)]

        SeedManager.set_global_seed(42)
        seeds_run2 = [SeedManager.get_worker_seed(i) for i in range(10)]

        assert seeds_run1 == seeds_run2

    def test_worker_operations_reproducible_with_worker_seed(self):
        """Test that operations using worker seeds are reproducible."""
        SeedManager.set_global_seed(42)

        def worker_operation(worker_id):
            seed = SeedManager.get_worker_seed(worker_id)
            np.random.seed(seed)
            return np.random.rand(5)

        # Run workers twice
        results1 = [worker_operation(i) for i in range(5)]

        SeedManager.set_global_seed(42)
        results2 = [worker_operation(i) for i in range(5)]

        for r1, r2 in zip(results1, results2):
            np.testing.assert_array_equal(r1, r2)

    def test_different_workers_produce_different_results(self):
        """Test that different workers produce different random sequences."""
        SeedManager.set_global_seed(42)

        def worker_operation(worker_id):
            seed = SeedManager.get_worker_seed(worker_id)
            np.random.seed(seed)
            return np.random.rand(5)

        worker0_result = worker_operation(0)
        worker1_result = worker_operation(1)

        assert not np.allclose(worker0_result, worker1_result)


class TestParallelProcessingReproducibility:
    """Tests for parallel processing reproducibility."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_bootstrap_worker_with_seed(self):
        """Test that bootstrap worker uses seed correctly."""
        from gridfia.utils.parallel_processing import _bootstrap_worker

        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        # Run with same seed twice
        result1 = _bootstrap_worker((group1, group2, {'seed': 42}))
        result2 = _bootstrap_worker((group1, group2, {'seed': 42}))

        assert result1['success']
        assert result2['success']
        assert result1['statistic'] == result2['statistic']

    def test_bootstrap_worker_different_seeds(self):
        """Test that bootstrap worker produces different results with different seeds."""
        from gridfia.utils.parallel_processing import _bootstrap_worker

        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        result1 = _bootstrap_worker((group1, group2, {'seed': 42}))
        result2 = _bootstrap_worker((group1, group2, {'seed': 123}))

        assert result1['success']
        assert result2['success']
        # Results should be different with different seeds
        assert result1['statistic'] != result2['statistic']

    def test_permutation_worker_with_seed(self):
        """Test that permutation worker uses seed correctly."""
        from gridfia.utils.parallel_processing import _permutation_worker

        combined = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n1, n2 = 5, 5

        # Run with same seed twice
        result1 = _permutation_worker((combined, n1, n2, 42))
        result2 = _permutation_worker((combined, n1, n2, 42))

        assert result1 == result2
        assert not np.isnan(result1)

    def test_permutation_worker_different_seeds(self):
        """Test that permutation worker produces different results with different seeds."""
        from gridfia.utils.parallel_processing import _permutation_worker

        combined = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n1, n2 = 5, 5

        result1 = _permutation_worker((combined, n1, n2, 42))
        result2 = _permutation_worker((combined, n1, n2, 123))

        # Results should be different with different seeds
        assert result1 != result2


class TestStatisticalAnalysisReproducibility:
    """Tests for statistical analysis reproducibility."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_permutation_test_reproducible(self):
        """Test that permutation test is reproducible with global seed."""
        import pandas as pd
        from gridfia.core.analysis.statistical_analysis import (
            StatisticalConfig,
            StatisticalTester,
        )

        config = StatisticalConfig(bootstrap_iterations=100)
        tester = StatisticalTester(config)

        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([3, 4, 5, 6, 7])

        SeedManager.set_global_seed(42)
        result1 = tester._permutation_test(group1, group2, n_permutations=100)

        SeedManager.set_global_seed(42)
        result2 = tester._permutation_test(group1, group2, n_permutations=100)

        assert result1['p_value'] == result2['p_value']
        assert result1['observed_difference'] == result2['observed_difference']

    def test_bootstrap_test_reproducible(self):
        """Test that bootstrap test is reproducible with global seed."""
        import pandas as pd
        from gridfia.core.analysis.statistical_analysis import (
            StatisticalConfig,
            StatisticalTester,
        )

        config = StatisticalConfig(bootstrap_iterations=100)
        tester = StatisticalTester(config)

        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([3, 4, 5, 6, 7])

        SeedManager.set_global_seed(42)
        result1 = tester._bootstrap_test(group1, group2)

        SeedManager.set_global_seed(42)
        result2 = tester._bootstrap_test(group1, group2)

        assert result1['difference_ci_lower'] == result2['difference_ci_lower']
        assert result1['difference_ci_upper'] == result2['difference_ci_upper']
        assert result1['significant'] == result2['significant']


class TestAPIReproducibility:
    """Tests for GridFIA API reproducibility."""

    def setup_method(self):
        """Reset SeedManager before each test."""
        SeedManager.reset()

    def teardown_method(self):
        """Reset SeedManager after each test."""
        SeedManager.reset()

    def test_api_seed_parameter(self):
        """Test that GridFIA API accepts seed parameter."""
        from gridfia.api import GridFIA

        api = GridFIA(seed=42)
        assert SeedManager.get_seed() == 42

    def test_api_set_seed_method(self):
        """Test that GridFIA API has set_seed method."""
        from gridfia.api import GridFIA

        api = GridFIA()
        assert SeedManager.get_seed() is None

        api.set_seed(123)
        assert SeedManager.get_seed() == 123

    def test_api_seed_persists_across_operations(self):
        """Test that seed persists across API operations."""
        from gridfia.api import GridFIA

        api = GridFIA(seed=42)

        # Verify seed is set
        assert SeedManager.get_seed() == 42

        # After operations, seed should still be set
        # (Note: we're just checking the seed manager state, not actual operations)
        result1 = np.random.rand(5)

        api.set_seed(42)  # Reset to same seed
        result2 = np.random.rand(5)

        np.testing.assert_array_equal(result1, result2)
