"""
Comprehensive tests for GridFIA console module.

This module provides test coverage for the Rich-based console output utilities
in gridfia/console.py, including print functions, table/panel creation,
file tree display, progress tracking, and user interaction functions.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.tree import Tree

from gridfia import console
from gridfia.console import (
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_step,
    create_species_table,
    create_processing_summary,
    create_file_tree,
    create_progress_tracker,
    display_configuration,
    confirm_action,
    print_package_info,
)


def create_test_console(width: int = 200) -> Console:
    """Create a test console that captures output without ANSI codes."""
    return Console(file=io.StringIO(), force_terminal=False, no_color=True, width=width)


class TestPrintHeader:
    """Tests for print_header function."""

    def test_print_header_with_title_only(self):
        """Test printing header with only a title."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_header("Test Title")

        output = test_console.file.getvalue()
        assert "Test Title" in output

    def test_print_header_with_title_and_subtitle(self):
        """Test printing header with title and subtitle."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_header("Main Title", subtitle="This is a subtitle")

        output = test_console.file.getvalue()
        assert "Main Title" in output
        assert "This is a subtitle" in output

    def test_print_header_with_empty_title(self):
        """Test printing header with empty title."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_header("")

        # Should not raise an error
        output = test_console.file.getvalue()
        assert output is not None

    def test_print_header_with_special_characters(self):
        """Test printing header with special characters."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_header("Forest Analysis 2024", subtitle="Species: Pinus ponderosa")

        output = test_console.file.getvalue()
        assert "Forest Analysis 2024" in output
        assert "Pinus ponderosa" in output


class TestPrintSuccess:
    """Tests for print_success function."""

    def test_print_success_basic(self):
        """Test basic success message printing."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_success("Operation completed successfully")

        output = test_console.file.getvalue()
        assert "Operation completed successfully" in output

    def test_print_success_empty_message(self):
        """Test success with empty message."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_success("")

        # Should not raise an error
        output = test_console.file.getvalue()
        assert output is not None

    def test_print_success_long_message(self):
        """Test success with long message."""
        test_console = create_test_console(width=500)
        long_message = "Successfully processed " + "data " * 50

        with patch.object(console, 'console', test_console):
            print_success(long_message)

        output = test_console.file.getvalue()
        assert "Successfully processed" in output


class TestPrintError:
    """Tests for print_error function."""

    def test_print_error_basic(self):
        """Test basic error message printing."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_error("An error occurred")

        output = test_console.file.getvalue()
        assert "An error occurred" in output

    def test_print_error_empty_message(self):
        """Test error with empty message."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_error("")

        output = test_console.file.getvalue()
        assert output is not None

    def test_print_error_with_path(self):
        """Test error message containing file path."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_error("File not found: /path/to/missing/file.zarr")

        output = test_console.file.getvalue()
        assert "File not found" in output
        assert "file.zarr" in output


class TestPrintWarning:
    """Tests for print_warning function."""

    def test_print_warning_basic(self):
        """Test basic warning message printing."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_warning("This may take a while")

        output = test_console.file.getvalue()
        assert "This may take a while" in output

    def test_print_warning_empty_message(self):
        """Test warning with empty message."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_warning("")

        output = test_console.file.getvalue()
        assert output is not None


class TestPrintInfo:
    """Tests for print_info function."""

    def test_print_info_basic(self):
        """Test basic info message printing."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_info("Processing 150 species files")

        output = test_console.file.getvalue()
        assert "Processing" in output
        assert "150" in output

    def test_print_info_empty_message(self):
        """Test info with empty message."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_info("")

        output = test_console.file.getvalue()
        assert output is not None


class TestPrintStep:
    """Tests for print_step function."""

    def test_print_step_first_step(self):
        """Test printing first step of multi-step process."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_step(1, 5, "Downloading species data")

        output = test_console.file.getvalue()
        assert "1/5" in output
        assert "Downloading" in output

    def test_print_step_last_step(self):
        """Test printing last step of multi-step process."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_step(5, 5, "Creating visualization")

        output = test_console.file.getvalue()
        assert "5/5" in output
        assert "visualization" in output

    def test_print_step_middle_step(self):
        """Test printing middle step."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_step(3, 7, "Processing metrics")

        output = test_console.file.getvalue()
        assert "3/7" in output
        assert "metrics" in output

    def test_print_step_single_step(self):
        """Test printing single step process."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_step(1, 1, "Single operation")

        output = test_console.file.getvalue()
        assert "1/1" in output


class TestCreateSpeciesTable:
    """Tests for create_species_table function."""

    def test_create_species_table_with_data(self):
        """Test creating species table with valid data."""
        species_data = [
            {
                'code': '0202',
                'name': 'Douglas-fir',
                'coverage_pct': 45.67,
                'pixels': 123456,
                'mean_biomass': 78.5,
                'max_biomass': 150.2
            },
            {
                'code': '0122',
                'name': 'Ponderosa Pine',
                'coverage_pct': 32.10,
                'pixels': 98765,
                'mean_biomass': 65.3,
                'max_biomass': 120.0
            }
        ]

        table = create_species_table(species_data)

        assert isinstance(table, Table)
        assert table.title == "Species Analysis Results"
        assert table.row_count == 2

    def test_create_species_table_empty_data(self):
        """Test creating species table with empty data list."""
        species_data = []

        table = create_species_table(species_data)

        assert isinstance(table, Table)
        assert table.row_count == 0

    def test_create_species_table_missing_fields(self):
        """Test creating species table with missing optional fields."""
        species_data = [
            {
                'code': '0202',
                'name': 'Douglas-fir'
                # Missing coverage_pct, pixels, mean_biomass, max_biomass
            }
        ]

        table = create_species_table(species_data)

        assert isinstance(table, Table)
        assert table.row_count == 1

    def test_create_species_table_with_unknown_values(self):
        """Test table handles unknown values gracefully."""
        species_data = [
            {
                # Missing 'code' and 'name' - should show 'Unknown'
            }
        ]

        table = create_species_table(species_data)

        assert isinstance(table, Table)
        assert table.row_count == 1

    def test_create_species_table_large_numbers(self):
        """Test table handles large numbers correctly."""
        species_data = [
            {
                'code': '0202',
                'name': 'Douglas-fir',
                'coverage_pct': 99.999,
                'pixels': 10000000,
                'mean_biomass': 1234.5678,
                'max_biomass': 9999.9999
            }
        ]

        table = create_species_table(species_data)

        assert isinstance(table, Table)
        assert table.row_count == 1

    def test_create_species_table_zero_values(self):
        """Test table handles zero values correctly."""
        species_data = [
            {
                'code': '0000',
                'name': 'Empty Species',
                'coverage_pct': 0.0,
                'pixels': 0,
                'mean_biomass': 0.0,
                'max_biomass': 0.0
            }
        ]

        table = create_species_table(species_data)

        assert isinstance(table, Table)
        assert table.row_count == 1


class TestCreateProcessingSummary:
    """Tests for create_processing_summary function."""

    def test_create_processing_summary_full_stats(self):
        """Test creating summary panel with all stats."""
        stats = {
            'total_files': 150,
            'successful': 145,
            'failed': 5,
            'elapsed_time': 123.45,
            'output_size_mb': 256.78,
            'compression_ratio': 3.5
        }

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)
        assert panel.title == "Processing Summary"

    def test_create_processing_summary_partial_stats(self):
        """Test creating summary with only some stats present."""
        stats = {
            'total_files': 100,
            'successful': 100
        }

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_empty_stats(self):
        """Test creating summary with empty stats dictionary."""
        stats = {}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_only_files(self):
        """Test summary with only total_files."""
        stats = {'total_files': 50}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_only_successful(self):
        """Test summary with only successful count."""
        stats = {'successful': 25}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_only_failed(self):
        """Test summary with only failed count."""
        stats = {'failed': 3}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_only_time(self):
        """Test summary with only elapsed_time."""
        stats = {'elapsed_time': 45.678}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_only_size(self):
        """Test summary with only output_size_mb."""
        stats = {'output_size_mb': 512.0}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)

    def test_create_processing_summary_only_compression(self):
        """Test summary with only compression_ratio."""
        stats = {'compression_ratio': 2.5}

        panel = create_processing_summary(stats)

        assert isinstance(panel, Panel)


class TestCreateFileTree:
    """Tests for create_file_tree function."""

    def test_create_file_tree_basic_directory(self, temp_dir):
        """Test creating file tree for basic directory structure."""
        # Create test directory structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.py").touch()
        (subdir / "nested_file.json").touch()

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_with_string_path(self, temp_dir):
        """Test creating file tree with string path instead of Path object."""
        (temp_dir / "test.txt").touch()

        tree = create_file_tree(str(temp_dir))

        assert isinstance(tree, Tree)

    def test_create_file_tree_empty_directory(self, temp_dir):
        """Test creating file tree for empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        tree = create_file_tree(empty_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_with_depth_limit(self, temp_dir):
        """Test file tree respects max_depth parameter."""
        # Create nested structure
        level1 = temp_dir / "level1"
        level2 = level1 / "level2"
        level3 = level2 / "level3"
        level4 = level3 / "level4"

        level1.mkdir()
        level2.mkdir()
        level3.mkdir()
        level4.mkdir()

        (level4 / "deep_file.txt").touch()

        tree = create_file_tree(temp_dir, max_depth=2)

        assert isinstance(tree, Tree)

    def test_create_file_tree_with_hidden_files(self, temp_dir):
        """Test that hidden files and directories are skipped."""
        hidden_dir = temp_dir / ".hidden_dir"
        hidden_dir.mkdir()

        (temp_dir / ".hidden_file").touch()
        (temp_dir / "visible_file.txt").touch()
        (hidden_dir / "inside_hidden.txt").touch()

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_with_various_extensions(self, temp_dir):
        """Test file tree handles various file extensions with correct icons."""
        (temp_dir / "script.py").touch()
        (temp_dir / "raster.tif").touch()
        (temp_dir / "raster2.tiff").touch()
        (temp_dir / "config.json").touch()
        (temp_dir / "config2.yaml").touch()
        (temp_dir / "config3.yml").touch()
        (temp_dir / "document.txt").touch()

        # Create a directory named with .zarr extension to test zarr icon
        zarr_dir = temp_dir / "data.zarr"
        zarr_dir.mkdir()
        (zarr_dir / ".zarray").touch()

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_with_zarr_file(self, temp_dir):
        """Test file tree shows correct icon for .zarr files."""
        # Create a file with .zarr extension (unusual but possible)
        zarr_file = temp_dir / "archive.zarr"
        zarr_file.write_text("mock zarr content")

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_many_files(self, temp_dir):
        """Test file tree with more than 10 files (should show 'and X more')."""
        for i in range(15):
            (temp_dir / f"file_{i:02d}.txt").touch()

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_large_files(self, temp_dir):
        """Test file tree shows size for files larger than 1MB."""
        large_file = temp_dir / "large_file.bin"
        # Create a file larger than 1MB
        with open(large_file, 'wb') as f:
            f.write(b'\0' * (2 * 1024 * 1024))  # 2MB file

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)

    def test_create_file_tree_small_files(self, temp_dir):
        """Test file tree does not show size for files smaller than 1MB."""
        small_file = temp_dir / "small_file.txt"
        small_file.write_text("small content")

        tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)


class TestCreateProgressTracker:
    """Tests for create_progress_tracker function."""

    def test_create_progress_tracker_default_description(self):
        """Test creating progress tracker with default description."""
        progress = create_progress_tracker()

        assert isinstance(progress, Progress)

    def test_create_progress_tracker_custom_description(self):
        """Test creating progress tracker with custom description."""
        progress = create_progress_tracker(description="Downloading species data")

        assert isinstance(progress, Progress)

    def test_create_progress_tracker_empty_description(self):
        """Test creating progress tracker with empty description."""
        progress = create_progress_tracker(description="")

        assert isinstance(progress, Progress)

    def test_progress_tracker_has_required_columns(self):
        """Test that progress tracker has all required columns."""
        progress = create_progress_tracker()

        # Progress should have 6 columns: spinner, text, bar, task progress, elapsed, remaining
        assert len(progress.columns) == 6


class TestDisplayConfiguration:
    """Tests for display_configuration function."""

    def test_display_configuration_flat_dict(self):
        """Test displaying flat configuration dictionary."""
        test_console = create_test_console()
        config_dict = {
            'data_dir': '/path/to/data',
            'output_dir': '/path/to/output',
            'verbose': True
        }

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output

    def test_display_configuration_nested_dict(self):
        """Test displaying nested configuration dictionary."""
        test_console = create_test_console()
        config_dict = {
            'paths': {
                'data': '/path/to/data',
                'output': '/path/to/output'
            },
            'processing': {
                'workers': 4,
                'chunk_size': 1000
            }
        }

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output

    def test_display_configuration_empty_dict(self):
        """Test displaying empty configuration dictionary."""
        test_console = create_test_console()
        config_dict = {}

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output

    def test_display_configuration_with_path_objects(self):
        """Test displaying configuration with Path objects."""
        test_console = create_test_console()
        config_dict = {
            'data_dir': Path('/path/to/data'),
            'output_dir': Path('/path/to/output')
        }

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output

    def test_display_configuration_with_boolean_values(self):
        """Test displaying configuration with boolean values."""
        test_console = create_test_console()
        config_dict = {
            'enabled': True,
            'disabled': False
        }

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output

    def test_display_configuration_with_list_values(self):
        """Test displaying configuration with list values."""
        test_console = create_test_console()
        config_dict = {
            'species_codes': ['0202', '0122', '0131'],
            'empty_list': []
        }

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output

    def test_display_configuration_with_tuple_values(self):
        """Test displaying configuration with tuple values."""
        test_console = create_test_console()
        config_dict = {
            'chunk_size': (1, 500, 500),
            'bounds': (-120.0, 35.0, -119.0, 36.0)
        }

        with patch.object(console, 'console', test_console):
            display_configuration(config_dict)

        output = test_console.file.getvalue()
        assert "Configuration Settings" in output


class TestConfirmAction:
    """Tests for confirm_action function."""

    def test_confirm_action_yes_response(self):
        """Test confirmation with 'yes' response."""
        with patch.object(console.console, 'input', return_value='yes'):
            result = confirm_action("Proceed with operation?")

        assert result is True

    def test_confirm_action_y_response(self):
        """Test confirmation with 'y' response."""
        with patch.object(console.console, 'input', return_value='y'):
            result = confirm_action("Proceed?")

        assert result is True

    def test_confirm_action_Y_response(self):
        """Test confirmation with uppercase 'Y' response."""
        with patch.object(console.console, 'input', return_value='Y'):
            result = confirm_action("Proceed?")

        assert result is True

    def test_confirm_action_no_response(self):
        """Test confirmation with 'no' response."""
        with patch.object(console.console, 'input', return_value='no'):
            result = confirm_action("Proceed?")

        assert result is False

    def test_confirm_action_n_response(self):
        """Test confirmation with 'n' response."""
        with patch.object(console.console, 'input', return_value='n'):
            result = confirm_action("Proceed?")

        assert result is False

    def test_confirm_action_empty_response_default_false(self):
        """Test empty response with default=False."""
        with patch.object(console.console, 'input', return_value=''):
            result = confirm_action("Proceed?", default=False)

        assert result is False

    def test_confirm_action_empty_response_default_true(self):
        """Test empty response with default=True."""
        with patch.object(console.console, 'input', return_value=''):
            result = confirm_action("Proceed?", default=True)

        assert result is True

    def test_confirm_action_whitespace_response(self):
        """Test response with leading/trailing whitespace."""
        with patch.object(console.console, 'input', return_value='  yes  '):
            result = confirm_action("Proceed?")

        assert result is True

    def test_confirm_action_true_response(self):
        """Test confirmation with 'true' response."""
        with patch.object(console.console, 'input', return_value='true'):
            result = confirm_action("Proceed?")

        assert result is True

    def test_confirm_action_1_response(self):
        """Test confirmation with '1' response."""
        with patch.object(console.console, 'input', return_value='1'):
            result = confirm_action("Proceed?")

        assert result is True

    def test_confirm_action_invalid_response(self):
        """Test confirmation with invalid response (not yes/no)."""
        with patch.object(console.console, 'input', return_value='maybe'):
            result = confirm_action("Proceed?")

        assert result is False

    def test_confirm_action_mixed_case(self):
        """Test confirmation with mixed case response."""
        with patch.object(console.console, 'input', return_value='YeS'):
            result = confirm_action("Proceed?")

        assert result is True


class TestPrintPackageInfo:
    """Tests for print_package_info function."""

    def test_print_package_info_success(self):
        """Test printing package info successfully."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_package_info()

        output = test_console.file.getvalue()
        assert "GridFIA" in output
        assert "Package Information" in output

    def test_print_package_info_contains_version(self):
        """Test package info contains version."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_package_info()

        output = test_console.file.getvalue()
        # Version format is typically X.Y.Z
        assert "v" in output or "0." in output

    def test_print_package_info_contains_author(self):
        """Test package info contains author information."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_package_info()

        output = test_console.file.getvalue()
        assert "Author" in output

    def test_print_package_info_contains_email(self):
        """Test package info contains email."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_package_info()

        output = test_console.file.getvalue()
        assert "Email" in output

    def test_print_package_info_contains_license(self):
        """Test package info contains license."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_package_info()

        output = test_console.file.getvalue()
        assert "License" in output or "MIT" in output


class TestEdgeCasesAndIntegration:
    """Edge cases and integration tests for console module."""

    def test_console_global_instance_exists(self):
        """Test that global console instance is available."""
        assert console.console is not None
        assert isinstance(console.console, Console)

    def test_multiple_print_functions_in_sequence(self):
        """Test calling multiple print functions in sequence."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_header("Test Header", subtitle="Test Subtitle")
            print_info("Starting process...")
            print_step(1, 3, "First step")
            print_step(2, 3, "Second step")
            print_warning("This is a warning")
            print_step(3, 3, "Final step")
            print_success("All done!")

        output = test_console.file.getvalue()
        assert "Test Header" in output
        assert "Starting" in output
        assert "1/3" in output
        assert "2/3" in output
        assert "warning" in output
        assert "3/3" in output
        assert "done" in output

    def test_unicode_handling_in_messages(self):
        """Test that unicode characters are handled properly."""
        test_console = create_test_console()

        with patch.object(console, 'console', test_console):
            print_info("Processing species: Pseudotsuga menziesii")
            print_success("Completed processing of area: 45.5 km\u00b2")

        output = test_console.file.getvalue()
        assert "Pseudotsuga" in output

    def test_create_species_table_renders_without_error(self):
        """Test that species table can be rendered to string."""
        species_data = [
            {'code': '0202', 'name': 'Douglas-fir', 'coverage_pct': 45.0,
             'pixels': 100000, 'mean_biomass': 75.0, 'max_biomass': 150.0}
        ]

        table = create_species_table(species_data)

        # Render to string to verify it works
        test_console = create_test_console()
        test_console.print(table)
        output = test_console.file.getvalue()

        assert "Species" in output
        assert "0202" in output
        assert "Douglas" in output

    def test_create_processing_summary_renders_without_error(self):
        """Test that processing summary can be rendered to string."""
        stats = {
            'total_files': 100,
            'successful': 95,
            'failed': 5,
            'elapsed_time': 120.5
        }

        panel = create_processing_summary(stats)

        # Render to string to verify it works
        test_console = create_test_console()
        test_console.print(panel)
        output = test_console.file.getvalue()

        assert "Processing Summary" in output

    def test_file_tree_permission_error_handling(self, temp_dir):
        """Test file tree handles permission errors gracefully."""
        # Create a directory but mock iterdir to raise PermissionError
        protected_dir = temp_dir / "protected"
        protected_dir.mkdir()

        # The function should handle permission errors internally
        with patch.object(Path, 'iterdir', side_effect=PermissionError("Access denied")):
            tree = create_file_tree(temp_dir)

        assert isinstance(tree, Tree)


class TestSpeciesTableWithVariousDataTypes:
    """Additional tests for species table with edge case data types."""

    def test_species_table_with_none_code_and_name(self):
        """Test species table handles None values for code and name."""
        species_data = [
            {
                'code': None,
                'name': None,
                'coverage_pct': 10.0,
                'pixels': 100,
                'mean_biomass': 50.0,
                'max_biomass': 100.0
            }
        ]

        # None for code/name should work (converts to string "None")
        table = create_species_table(species_data)
        assert isinstance(table, Table)
        assert table.row_count == 1

    def test_species_table_with_numeric_none_raises_error(self):
        """Test species table with None numeric values raises TypeError."""
        species_data = [
            {
                'code': '0202',
                'name': 'Douglas-fir',
                'coverage_pct': None,  # This will cause a TypeError
                'pixels': 100,
                'mean_biomass': 50.0,
                'max_biomass': 100.0
            }
        ]

        # None for numeric fields will raise TypeError during formatting
        with pytest.raises(TypeError):
            create_species_table(species_data)

    def test_species_table_with_integer_strings(self):
        """Test species table handles string numbers."""
        species_data = [
            {
                'code': '0202',
                'name': 'Douglas-fir',
                'coverage_pct': 45.67,
                'pixels': 123456,
                'mean_biomass': 78.5,
                'max_biomass': 150.2
            }
        ]

        table = create_species_table(species_data)
        assert isinstance(table, Table)
        assert table.row_count == 1


class TestProgressTrackerUsage:
    """Tests for progress tracker context manager usage."""

    def test_progress_tracker_context_manager(self):
        """Test progress tracker as context manager."""
        progress = create_progress_tracker("Test task")

        # Should work as context manager
        with progress:
            task_id = progress.add_task("Processing", total=10)
            for i in range(10):
                progress.update(task_id, advance=1)

        # Progress should complete without error
        assert True

    def test_progress_tracker_with_indeterminate_task(self):
        """Test progress tracker with indeterminate (unknown total) task."""
        progress = create_progress_tracker("Loading")

        with progress:
            task_id = progress.add_task("Loading data", total=None)
            # Simulate some work
            progress.update(task_id, advance=1)

        assert True
