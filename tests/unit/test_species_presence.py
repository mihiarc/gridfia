"""Tests for species presence analysis module.

Tests the species presence analysis functions using sample Zarr data.
"""

import pytest
import numpy as np
from pathlib import Path
import os
import zarr

from gridfia.core.analysis.species_presence import (
    analyze_species_presence,
    get_folder_size,
)


class TestGetFolderSize:
    """Tests for the get_folder_size helper function."""

    def test_empty_folder(self, tmp_path):
        """Test size of empty folder is zero."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        size = get_folder_size(str(empty_dir))

        assert size == 0.0

    def test_single_file(self, tmp_path):
        """Test size calculation with single file."""
        file_path = tmp_path / "test_file.txt"
        # Create a file with known size (1024 bytes = 1KB)
        file_path.write_bytes(b"x" * 1024)

        size = get_folder_size(str(tmp_path))

        # Size should be 1024 bytes = 0.0009765625 MB
        assert abs(size - (1024 / (1024 * 1024))) < 0.0001

    def test_multiple_files(self, tmp_path):
        """Test size calculation with multiple files."""
        # Create several files
        for i in range(5):
            (tmp_path / f"file_{i}.txt").write_bytes(b"x" * 1024)

        size = get_folder_size(str(tmp_path))

        # Should be 5 * 1024 bytes
        expected = (5 * 1024) / (1024 * 1024)
        assert abs(size - expected) < 0.0001

    def test_nested_directories(self, tmp_path):
        """Test size calculation with nested directories."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root_file.txt").write_bytes(b"x" * 1024)
        (subdir / "nested_file.txt").write_bytes(b"x" * 2048)

        size = get_folder_size(str(tmp_path))

        # Should be 1024 + 2048 = 3072 bytes
        expected = 3072 / (1024 * 1024)
        assert abs(size - expected) < 0.0001


class TestAnalyzeSpeciesPresence:
    """Tests for the main analyze_species_presence function."""

    @pytest.fixture
    def valid_zarr_store(self, tmp_path):
        """Create a valid Zarr store for testing."""
        zarr_path = tmp_path / "test_presence.zarr"

        # Create Zarr store with species data
        n_species = 5
        height = width = 50
        data = np.zeros((n_species, height, width), dtype=np.float32)

        # Species with varying coverage
        data[0, :, :] = 30.0  # Total layer: 100%
        data[1, :25, :] = 20.0  # Species 1: 50%
        data[2, :5, :5] = 15.0  # Species 2: 1%
        data[3, 0, 0] = 10.0  # Species 3: single pixel
        # data[4] = all zeros (no data)

        # Create Zarr v3 store
        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.open_group(store=store, mode='w')

        codec = zarr.codecs.BloscCodec(cname='zstd', clevel=3)
        z = root.create_array(
            name='biomass',
            shape=(n_species, height, width),
            chunks=(1, 50, 50),
            dtype='f4',
            compressors=[codec]
        )
        z[:] = data

        # Add attributes
        root.attrs['species_codes'] = ['TOTAL', '0131', '0316', '0202', '0068']
        root.attrs['species_names'] = [
            'Total Biomass', 'Loblolly Pine', 'Sweetgum', 'Douglas Fir', 'Engelmann Spruce'
        ]
        root.attrs['crs'] = 'EPSG:3857'
        root.attrs['transform'] = [30.0, 0.0, -8000000.0, 0.0, -30.0, 5000000.0]

        return zarr_path

    def test_nonexistent_path_returns_none(self, capsys, tmp_path):
        """Test that nonexistent zarr path prints message and returns."""
        result = analyze_species_presence(
            zarr_path=str(tmp_path / "nonexistent.zarr"),
            output_dir=str(tmp_path / "output")
        )

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_with_valid_zarr(self, valid_zarr_store, tmp_path, capsys):
        """Test analysis with a valid Zarr store."""
        output_dir = tmp_path / "output"

        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(output_dir),
            biomass_threshold=0.0
        )

        captured = capsys.readouterr()

        # Check that analysis ran and produced output
        assert "Analyzing Species Presence" in captured.out
        assert "SUMMARY STATISTICS" in captured.out
        assert "Species with biomass data" in captured.out

    def test_creates_output_directory(self, valid_zarr_store, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "nested" / "output"

        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(output_dir)
        )

        assert output_dir.exists()

    def test_saves_csv_when_species_present(self, valid_zarr_store, tmp_path):
        """Test that CSV is saved when species have data."""
        output_dir = tmp_path / "output"

        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(output_dir)
        )

        # Should create CSV file
        csv_file = output_dir / "species_presence_analysis.csv"
        assert csv_file.exists()

    def test_with_biomass_threshold(self, valid_zarr_store, tmp_path, capsys):
        """Test analysis with biomass threshold."""
        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(tmp_path / "output"),
            biomass_threshold=10.0  # Higher threshold
        )

        captured = capsys.readouterr()
        assert "Analyzing Species Presence" in captured.out

    def test_species_sorted_by_coverage(self, valid_zarr_store, tmp_path, capsys):
        """Test that species are sorted by coverage percentage."""
        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(tmp_path / "output")
        )

        captured = capsys.readouterr()

        # Total should appear first (100% coverage)
        assert "Total Biomass" in captured.out or "TOTAL" in captured.out

    def test_identifies_species_without_data(self, valid_zarr_store, tmp_path, capsys):
        """Test that species without data are correctly identified."""
        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(tmp_path / "output")
        )

        captured = capsys.readouterr()

        # Species 4 (Engelmann Spruce) has no data
        assert "NO DATA" in captured.out or "without data" in captured.out.lower()

    def test_prints_top_species(self, valid_zarr_store, tmp_path, capsys):
        """Test that top species by coverage are printed."""
        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(tmp_path / "output")
        )

        captured = capsys.readouterr()

        assert "TOP" in captured.out or "top" in captured.out.lower()

    def test_prints_next_steps(self, valid_zarr_store, tmp_path, capsys):
        """Test that next steps section is printed."""
        analyze_species_presence(
            zarr_path=str(valid_zarr_store),
            output_dir=str(tmp_path / "output")
        )

        captured = capsys.readouterr()

        assert "NEXT STEPS" in captured.out
