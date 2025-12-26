#!/usr/bin/env python3
"""
Smoke tests for GridFIA examples.

These tests verify that each example can be imported and runs without errors.
They don't test the full functionality, just that the examples are syntactically
correct and can execute their main functions.
"""

import sys
import subprocess
from pathlib import Path
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Get the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class TestExamplesSmoke:
    """Smoke tests to ensure examples run without errors."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and cleanup for each test."""
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = Path.cwd()

        yield

        # Cleanup
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_quickstart_imports(self):
        """Test that 01_quickstart.py can be imported."""
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            # Mock the API to avoid actual downloads
            with patch('gridfia.GridFIA') as mock_api:
                mock_instance = MagicMock()
                mock_api.return_value = mock_instance

                # Import should work without errors
                # Use importlib to handle module name starting with digit
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "quickstart",
                    EXAMPLES_DIR / "01_quickstart.py"
                )
                quickstart = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(quickstart)
                assert hasattr(quickstart, 'main')
        finally:
            sys.path.pop(0)

    def test_api_overview_imports(self):
        """Test that 02_api_overview.py can be imported."""
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            with patch('gridfia.GridFIA') as mock_api:
                # Use importlib to handle module name starting with digit
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "api_overview",
                    EXAMPLES_DIR / "02_api_overview.py"
                )
                api_overview = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(api_overview)

                # Check that all example functions exist
                assert hasattr(api_overview, 'example_1_list_species')
                assert hasattr(api_overview, 'example_2_location_config')
        finally:
            sys.path.pop(0)

    def test_location_configs_imports(self):
        """Test that 03_location_configs.py can be imported."""
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            # Use importlib to handle module name starting with digit
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "location_configs",
                EXAMPLES_DIR / "03_location_configs.py"
            )
            location_configs = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(location_configs)

            # Check main functions exist
            assert hasattr(location_configs, 'create_state_configs')
            assert hasattr(location_configs, 'create_county_configs')
        finally:
            sys.path.pop(0)

    def test_calculations_imports(self):
        """Test that 04_calculations.py can be imported."""
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            # Use importlib to handle module name starting with digit
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "calculations",
                EXAMPLES_DIR / "04_calculations.py"
            )
            calculations = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(calculations)

            # Check functions exist (actual function names in the example)
            assert hasattr(calculations, 'show_available_calculations')
            assert hasattr(calculations, 'example_basic_calculations')
        finally:
            sys.path.pop(0)

    def test_species_analysis_imports(self):
        """Test that 05_species_analysis.py can be imported."""
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            # Use importlib to handle module name starting with digit
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "species_analysis",
                EXAMPLES_DIR / "05_species_analysis.py"
            )
            species_analysis = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(species_analysis)

            # Check functions exist
            assert hasattr(species_analysis, 'analyze_species_proportions')
            assert hasattr(species_analysis, 'analyze_southern_yellow_pine')
        finally:
            sys.path.pop(0)

    def test_wake_county_full_imports(self):
        """Test that 06_wake_county_full.py can be imported."""
        sys.path.insert(0, str(EXAMPLES_DIR.parent))
        try:
            # Use importlib to handle module name starting with digit
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "wake_county",
                EXAMPLES_DIR / "06_wake_county_full.py"
            )
            wake_county = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(wake_county)

            # Check main functions exist (actual function names in the example)
            assert hasattr(wake_county, 'download_wake_county_data')
            assert hasattr(wake_county, 'create_wake_zarr')
            assert hasattr(wake_county, 'run_comprehensive_calculations')
        finally:
            sys.path.pop(0)

    @pytest.mark.parametrize("example_file", [
        "01_quickstart.py",
        "02_api_overview.py",
        "03_location_configs.py",
        "04_calculations.py",
        "05_species_analysis.py",
        "06_wake_county_full.py"
    ])
    def test_example_syntax(self, example_file):
        """Test that each example file has valid Python syntax."""
        example_path = EXAMPLES_DIR / example_file
        assert example_path.exists(), f"Example file {example_file} not found"

        # Compile the file to check for syntax errors
        with open(example_path, 'r') as f:
            code = f.read()

        try:
            compile(code, str(example_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {example_file}: {e}")

    def test_utils_module(self):
        """Test that the utils module can be imported from gridfia.examples."""
        from gridfia.examples import (
            AnalysisConfig,
            cleanup_example_outputs,
            safe_download_species,
            safe_load_zarr_with_memory_check,
            create_zarr_from_rasters,
            create_sample_zarr,
            print_zarr_info,
            calculate_basic_stats,
            validate_species_codes
        )

        # Check that AnalysisConfig has expected attributes
        config = AnalysisConfig()
        assert hasattr(config, 'biomass_threshold')
        assert hasattr(config, 'diversity_percentile')
        assert hasattr(config, 'max_pixels')

        # Verify defaults
        assert config.biomass_threshold == 1.0
        assert config.diversity_percentile == 90
        assert config.max_pixels == 1_000_000

    @pytest.mark.skipif(not EXAMPLES_DIR.exists(), reason="Examples directory not found")
    def test_example_readme_exists(self):
        """Test that examples README exists."""
        readme_path = EXAMPLES_DIR / "README.md"
        assert readme_path.exists(), "Examples README.md not found"

        # Check that README has expected content
        with open(readme_path, 'r') as f:
            content = f.read()

        assert "BigMap Examples" in content or "GridFIA Examples" in content
        assert "01_quickstart.py" in content
        assert "Getting Started" in content

    def test_no_hardcoded_paths(self):
        """Test that examples don't use hardcoded absolute paths."""
        bad_patterns = [
            "/Users/",
            "/home/",
            "C:\\Users\\",
            "D:\\",
            "/tmp/specific_user_dir"
        ]

        for example_file in EXAMPLES_DIR.glob("*.py"):
            if example_file.name == "__init__.py":
                continue

            with open(example_file, 'r') as f:
                content = f.read()

            for pattern in bad_patterns:
                assert pattern not in content, \
                    f"Hardcoded path '{pattern}' found in {example_file.name}"

    def test_cleanup_function_works(self):
        """Test that the cleanup function works correctly."""
        from gridfia.examples import cleanup_example_outputs

        # Create test directories
        test_dirs = ["test_quickstart_data", "test_configs", "test_output"]
        for dir_name in test_dirs:
            test_path = Path(self.temp_dir) / dir_name
            test_path.mkdir(parents=True)
            assert test_path.exists()

        # Change to temp directory and run cleanup
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            cleanup_example_outputs(test_dirs)

            # Verify directories were removed
            for dir_name in test_dirs:
                test_path = Path(self.temp_dir) / dir_name
                assert not test_path.exists()
        finally:
            os.chdir(old_cwd)


class TestExampleIntegration:
    """Integration tests for example functionality."""

    def test_analysis_config_usage(self):
        """Test that AnalysisConfig works as expected."""
        from gridfia.examples import AnalysisConfig

        # Test default config
        config = AnalysisConfig()
        assert config.biomass_threshold == 1.0
        assert config.chunk_size == (1, 1000, 1000)

        # Test custom config
        custom_config = AnalysisConfig(
            biomass_threshold=2.0,
            diversity_percentile=95,
            max_pixels=500_000
        )
        assert custom_config.biomass_threshold == 2.0
        assert custom_config.diversity_percentile == 95
        assert custom_config.max_pixels == 500_000

    def test_safe_download_with_mock(self):
        """Test safe_download_species with mocked API."""
        from gridfia.examples import safe_download_species

        mock_api = MagicMock()
        mock_api.download_species.return_value = [Path("test1.tif"), Path("test2.tif")]

        # Should succeed on first try
        result = safe_download_species(
            mock_api,
            state="Test State",
            species_codes=["0001", "0002"]
        )

        assert len(result) == 2
        assert mock_api.download_species.called

    def test_safe_download_with_retry(self):
        """Test that safe_download_species retries on failure."""
        from gridfia.examples import safe_download_species

        mock_api = MagicMock()
        # Fail twice, then succeed
        mock_api.download_species.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            [Path("test.tif")]
        ]

        result = safe_download_species(
            mock_api,
            state="Test State",
            species_codes=["0001"],
            max_retries=3
        )

        assert len(result) == 1
        assert mock_api.download_species.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])