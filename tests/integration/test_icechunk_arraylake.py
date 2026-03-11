"""
Integration tests for ZarrStore.from_icechunk() with the real Arraylake datacube.

These tests connect to the ctrees/sfi_gridfia repo on Arraylake (branch: mihiar)
and verify that ZarrStore correctly reads the BIGMAP 4D datacube.

Requires:
    - ARRAYLAKE_TOKEN environment variable set with a valid API token
    - Network access to Arraylake

Run with:
    ARRAYLAKE_TOKEN=... uv run pytest tests/integration/test_icechunk_arraylake.py -v
"""

import os

import numpy as np
import pytest

try:
    import icechunk
    from arraylake import Client

    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

from gridfia.utils.zarr_utils import ZarrStore

ARRAYLAKE_TOKEN = os.environ.get("ARRAYLAKE_TOKEN")
REPO_NAME = "ctrees/sfi_gridfia"
BRANCH = "mihiar"
YEAR = 2018

skip_no_icechunk = pytest.mark.skipif(
    not HAS_ICECHUNK, reason="icechunk/arraylake packages not installed"
)
skip_no_token = pytest.mark.skipif(
    not ARRAYLAKE_TOKEN, reason="ARRAYLAKE_TOKEN not set"
)

pytestmark = [skip_no_icechunk, skip_no_token]


@pytest.fixture(scope="module")
def arraylake_repo():
    """Open the real Arraylake repo once for all tests in this module."""
    client = Client(token=ARRAYLAKE_TOKEN)
    return client.get_repo(REPO_NAME)


@pytest.fixture(scope="module")
def icechunk_store(arraylake_repo):
    """Open a ZarrStore from the real Arraylake datacube."""
    return ZarrStore.from_icechunk(arraylake_repo, branch=BRANCH, year=YEAR)


class TestArraylakeConnection:
    """Verify basic connectivity and datacube structure."""

    def test_store_opens_successfully(self, icechunk_store):
        assert icechunk_store is not None
        assert not icechunk_store._closed

    def test_shape_is_3d(self, icechunk_store):
        shape = icechunk_store.shape
        assert len(shape) == 3
        assert shape[0] >= 2  # At least 2 species (Loblolly + Total)
        assert shape[1] > 0
        assert shape[2] > 0

    def test_expected_dimensions(self, icechunk_store):
        """Datacube should be full CONUS at 30m resolution."""
        shape = icechunk_store.shape
        # From issue: (84877, 164319) spatial dims
        assert shape[1] == 84877
        assert shape[2] == 164319


class TestSpeciesMetadata:
    """Verify species codes and names are correctly mapped."""

    def test_species_codes_populated(self, icechunk_store):
        codes = icechunk_store.species_codes
        assert len(codes) >= 2
        assert all(isinstance(c, str) for c in codes)

    def test_expected_species_present(self, icechunk_store):
        codes = icechunk_store.species_codes
        assert "0131" in codes  # Loblolly Pine
        assert "0000" in codes  # Total

    def test_species_names_populated(self, icechunk_store):
        names = icechunk_store.species_names
        assert len(names) >= 2
        assert "Loblolly Pine" in names
        assert "Total" in names

    def test_codes_and_names_same_length(self, icechunk_store):
        assert len(icechunk_store.species_codes) == len(icechunk_store.species_names)

    def test_get_species_index(self, icechunk_store):
        idx = icechunk_store.get_species_index("0131")
        assert isinstance(idx, int)
        assert 0 <= idx < icechunk_store.num_species

    def test_get_species_info(self, icechunk_store):
        info = icechunk_store.get_species_info()
        assert len(info) >= 2
        codes_in_info = [entry["code"] for entry in info]
        assert "0131" in codes_in_info


class TestSpatialMetadata:
    """Verify CRS, transform, and bounds are correctly reconstructed."""

    def test_crs_is_epsg_5070(self, icechunk_store):
        assert str(icechunk_store.crs) == "EPSG:5070"

    def test_transform_resolution_is_30m(self, icechunk_store):
        transform = icechunk_store.transform
        assert transform.a == 30.0   # x pixel size
        assert transform.e == -30.0  # y pixel size (north-up)

    def test_transform_has_valid_origin(self, icechunk_store):
        transform = icechunk_store.transform
        # EPSG:5070 CONUS origin should be in the negative x range
        assert transform.c < 0  # origin x (west of center)
        assert transform.f > 0  # origin y (north)

    def test_bounds_are_valid(self, icechunk_store):
        bounds = icechunk_store.bounds
        assert len(bounds) == 4
        left, bottom, right, top = bounds
        assert left < right
        assert bottom < top

    def test_extent_for_matplotlib(self, icechunk_store):
        extent = icechunk_store.get_extent()
        left, right, bottom, top = extent
        assert left < right
        assert bottom < top


class TestDataAccess:
    """Verify actual biomass data can be read from the datacube."""

    def test_read_spatial_chunk(self, icechunk_store):
        """Read a small spatial chunk — southeastern US where Loblolly grows."""
        chunk = icechunk_store.biomass[0, 60000:60010, 110000:110010]
        assert chunk.shape == (10, 10)
        assert chunk.dtype == np.float32

    def test_data_has_valid_values(self, icechunk_store):
        """Total biomass layer should have real values in the SE US."""
        total_idx = icechunk_store.get_species_index("0000")
        chunk = icechunk_store.biomass[total_idx, 60000:60010, 110000:110010]
        valid = chunk[~np.isnan(chunk)]
        assert len(valid) > 0, "Expected non-NaN values in southeastern CONUS"
        assert np.all(valid >= 0), "Biomass values should be non-negative"

    def test_get_species_layer_reads_data(self, icechunk_store):
        """get_species_layer should return a 2D slice for Loblolly Pine."""
        # Read just a small window via the full method path
        idx = icechunk_store.get_species_index("0131")
        chunk = icechunk_store.biomass[idx, 60000:60010, 110000:110010]
        assert chunk.shape == (10, 10)
        assert chunk.dtype == np.float32

    def test_summary_returns_valid_dict(self, icechunk_store):
        summary = icechunk_store.summary()
        assert summary["shape"] == (2, 84877, 164319)
        assert summary["crs"] == "EPSG:5070"
        assert summary["path"] is None  # Icechunk has no local path
        assert "0131" in summary["species_codes"]

    def test_repr_shows_icechunk(self, icechunk_store):
        repr_str = repr(icechunk_store)
        assert "icechunk" in repr_str
        assert "year=2018" in repr_str
