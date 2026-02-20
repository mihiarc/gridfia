"""
End-to-end integration tests for GridFIA using real FIA BIGMAP API data.

These tests exercise the full pipeline:
1. API discovery (list calculations, list species)
2. Download species data from the FIA BIGMAP REST API
3. Create a Zarr store from downloaded GeoTIFFs
4. Validate the Zarr store structure
5. ZarrStore public API (read, slice, query species)
6. Run all parameter-free calculations
7. Run statistical calculations with bootstrap CIs
8. Create visualization maps

All tests use real API calls as required by project guidelines. The test
suite downloads 3 species for a ~5km x 5km area near Umstead State Park, NC
(~30m resolution) and creates a local Zarr store for subsequent tests.

Runtime: ~10-15 minutes total (most time spent on FIA API downloads).

Usage:
    # Run all e2e tests
    uv run pytest tests/integration/test_end_to_end.py -v -s

    # Quick smoke test (discovery only, ~5s)
    uv run pytest tests/integration/test_end_to_end.py::TestDiscovery -v

    # Run only calculation tests
    uv run pytest tests/integration/test_end_to_end.py::TestAllCalculations -v
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless rendering before any pyplot import

import numpy as np
import pytest

from gridfia import GridFIA
from gridfia.utils.zarr_utils import ZarrStore
from gridfia.exceptions import (
    CalculationFailed,
    SpeciesNotFound,
)


# ---------------------------------------------------------------------------
# Session-scoped Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api():
    """Session-scoped GridFIA instance (no seed — default)."""
    return GridFIA()


@pytest.fixture(scope="session")
def session_temp_dir(tmp_path_factory):
    """Session-scoped temporary directory shared across all tests."""
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="session")
def local_zarr_path(api, session_temp_dir):
    """Create a local Zarr store by downloading 3 species from FIA API.

    Downloads Red maple (0316), Loblolly pine (0131), and Eastern white pine (0261)
    for a ~5km x 5km area near Umstead State Park, NC at 30m resolution.
    Then creates a Zarr store from the GeoTIFFs.
    """
    download_dir = session_temp_dir / "downloads"
    download_dir.mkdir(exist_ok=True)
    zarr_path = session_temp_dir / "local_test.zarr"

    # Web Mercator bbox: ~5km x 5km near Umstead State Park, NC
    bbox_wm = (-8765000, 4270000, -8760000, 4275000)
    species_codes = ["0316", "0131", "0261"]

    files = api.download_species(
        output_dir=download_dir,
        species_codes=species_codes,
        bbox=bbox_wm,
        crs="102100",
    )

    if len(files) < 2:
        pytest.skip("Not enough species downloaded from FIA API for Zarr creation")

    result = api.create_zarr(download_dir, zarr_path)
    assert result == zarr_path
    assert zarr_path.exists()
    return zarr_path


@pytest.fixture
def temp_dir(tmp_path):
    """Per-test temporary directory for output isolation."""
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    """Tests for listing available calculations and species (~10s)."""

    def test_list_all_calculations(self, api):
        """list_calculations returns all 15 registered calculation names."""
        calcs = api.list_calculations()

        assert isinstance(calcs, list)
        assert len(calcs) == 15

        expected_subset = {
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "evenness",
            "total_biomass",
            "total_biomass_comparison",
            "dominant_species",
            "species_presence",
            "species_dominance",
            "rare_species",
            "common_species",
            "species_proportion",
            "species_percentage",
            "species_group_proportion",
            "biomass_threshold",
        }
        assert set(calcs) == expected_subset

    def test_list_species(self, api):
        """list_species returns > 100 species from the FIA BIGMAP API."""
        from gridfia.api import SpeciesInfo

        species = api.list_species()
        assert isinstance(species, list)
        assert len(species) > 100

        # Verify structure
        first = species[0]
        assert isinstance(first, SpeciesInfo)
        assert first.species_code is not None
        assert first.common_name is not None
        assert first.scientific_name is not None

    def test_known_species_present(self, api):
        """Known common species (Red maple, Douglas-fir, E. white pine) are in the API."""
        species = api.list_species()
        codes = {s.species_code for s in species}

        known_species = {
            "0316": "Red maple",
            "0202": "Douglas-fir",
            "0261": "Eastern white pine",
            "0131": "Loblolly pine",
        }
        for code, name in known_species.items():
            assert code in codes, f"{name} ({code}) not found in FIA species list"


# ---------------------------------------------------------------------------
# 2. Download Species (FIA BIGMAP API)
# ---------------------------------------------------------------------------

class TestDownloadSpecies:
    """Tests for downloading species data from FIA API (~2 min)."""

    def test_download_with_web_mercator_bbox(self, api, temp_dir):
        """Download 1 species with Web Mercator bbox produces valid GeoTIFF."""
        import rasterio

        bbox_wm = (-8765000, 4270000, -8760000, 4275000)
        files = api.download_species(
            output_dir=temp_dir,
            species_codes=["0316"],  # Red maple
            bbox=bbox_wm,
            crs="102100",
        )

        assert isinstance(files, list)
        assert len(files) >= 1
        assert files[0].exists()
        assert files[0].stat().st_size > 0

        with rasterio.open(files[0]) as src:
            assert src.width > 0
            assert src.height > 0
            assert src.crs is not None

    def test_download_with_wgs84_bbox(self, api, temp_dir):
        """Download with WGS84 coordinates triggers automatic CRS transformation."""
        bbox_wgs84 = (-79.1, 35.75, -79.05, 35.8)

        files = api.download_species(
            output_dir=temp_dir,
            species_codes=["0316"],  # Red maple
            bbox=bbox_wgs84,
            crs="EPSG:4326",
        )

        assert isinstance(files, list)
        assert len(files) >= 1
        assert files[0].exists()


# ---------------------------------------------------------------------------
# 3. Create and Validate Zarr
# ---------------------------------------------------------------------------

class TestCreateAndValidateZarr:
    """Tests for Zarr creation and validation (~30s)."""

    def test_zarr_created_successfully(self, local_zarr_path):
        """The session fixture created a valid Zarr store."""
        assert local_zarr_path.exists()
        assert ZarrStore.is_valid_store(local_zarr_path)

    def test_validate_zarr_structure(self, api, local_zarr_path):
        """validate_zarr returns dict with correct structure."""
        info = api.validate_zarr(local_zarr_path)
        assert isinstance(info, dict)
        assert "shape" in info
        assert "num_species" in info
        assert info["num_species"] >= 3  # At least 3 species + total
        assert "species" in info
        assert isinstance(info["species"], list)

    def test_validate_invalid_path(self, api, temp_dir):
        """validate_zarr on nonexistent path raises an exception."""
        with pytest.raises(Exception):
            api.validate_zarr(temp_dir / "does_not_exist.zarr")


# ---------------------------------------------------------------------------
# 4. ZarrStore API
# ---------------------------------------------------------------------------

class TestZarrStoreAPI:
    """Tests for ZarrStore public interface (~30s)."""

    def test_from_path(self, local_zarr_path):
        """from_path returns correct path, shape, height, width."""
        store = ZarrStore.from_path(local_zarr_path)
        assert store.path == local_zarr_path
        assert len(store.shape) == 3
        assert store.height == store.shape[1]
        assert store.width == store.shape[2]

    def test_context_manager(self, local_zarr_path):
        """Works in a with block; data is accessible inside."""
        with ZarrStore.open(local_zarr_path) as store:
            assert store.shape[0] >= 3
            data = store.biomass[0, :5, :5]
            assert np.asarray(data).shape == (5, 5)

    def test_get_species_layer(self, local_zarr_path):
        """get_species_layer for Red maple (0316) returns correct 2D array."""
        store = ZarrStore.from_path(local_zarr_path)
        layer = store.get_species_layer("0316")
        assert layer.ndim == 2
        assert layer.shape == (store.height, store.width)

    def test_get_species_layer_not_found(self, local_zarr_path):
        """get_species_layer for nonexistent code raises SpeciesNotFound."""
        store = ZarrStore.from_path(local_zarr_path)
        with pytest.raises(SpeciesNotFound):
            store.get_species_layer("9999")

    def test_get_species_info(self, local_zarr_path):
        """get_species_info returns list of dicts with index, code, name keys."""
        store = ZarrStore.from_path(local_zarr_path)
        species_info = store.get_species_info()
        assert isinstance(species_info, list)
        assert len(species_info) > 0
        for sp in species_info:
            assert "index" in sp
            assert "code" in sp
            assert "name" in sp

    def test_get_extent(self, local_zarr_path):
        """get_extent returns 4-tuple with left != right and bottom != top."""
        store = ZarrStore.from_path(local_zarr_path)
        extent = store.get_extent()
        assert len(extent) == 4
        left, right, bottom, top = extent
        assert left != right
        assert bottom != top

    def test_summary(self, local_zarr_path):
        """summary returns dict with expected keys."""
        store = ZarrStore.from_path(local_zarr_path)
        summary = store.summary()
        assert isinstance(summary, dict)
        for key in ("shape", "num_species", "crs", "species_codes", "dtype"):
            assert key in summary, f"Missing key in summary: {key}"


# ---------------------------------------------------------------------------
# 5. All Calculations
# ---------------------------------------------------------------------------

class TestAllCalculations:
    """Tests for calculate_metrics with real Zarr data (~2 min)."""

    def test_diversity_calculations(self, api, local_zarr_path, temp_dir):
        """Richness, Shannon, Simpson, Evenness produce valid GeoTIFF output."""
        import rasterio

        calcs = ["species_richness", "shannon_diversity", "simpson_diversity", "evenness"]
        results = api.calculate_metrics(
            local_zarr_path,
            calculations=calcs,
            output_dir=temp_dir / "diversity",
        )

        assert len(results) == len(calcs)

        for result in results:
            assert result.output_path.exists(), f"{result.name} output missing"
            assert result.output_path.stat().st_size > 0

            with rasterio.open(result.output_path) as src:
                data = src.read(1)
                assert data.shape[0] > 0 and data.shape[1] > 0

                if result.name == "species_richness":
                    valid = data[np.isfinite(data)]
                    assert valid.min() >= 0
                elif result.name in ("shannon_diversity", "simpson_diversity"):
                    valid = data[np.isfinite(data)]
                    assert valid.min() >= 0

    def test_biomass_calculations(self, api, local_zarr_path, temp_dir):
        """Total biomass and comparison produce valid outputs."""
        import rasterio

        calcs = ["total_biomass", "total_biomass_comparison"]
        results = api.calculate_metrics(
            local_zarr_path,
            calculations=calcs,
            output_dir=temp_dir / "biomass",
        )

        assert len(results) == len(calcs)

        for result in results:
            assert result.output_path.exists()

            with rasterio.open(result.output_path) as src:
                data = src.read(1)
                if result.name == "total_biomass":
                    valid = data[np.isfinite(data)]
                    assert valid.min() >= 0, "Biomass should be non-negative"

    def test_species_calculations(self, api, local_zarr_path, temp_dir):
        """Dominant, rare, common species produce valid outputs."""
        calcs = ["dominant_species", "rare_species", "common_species"]
        results = api.calculate_metrics(
            local_zarr_path,
            calculations=calcs,
            output_dir=temp_dir / "species",
        )

        assert len(results) == len(calcs)

        for result in results:
            assert result.output_path.exists()
            assert result.output_path.stat().st_size > 0

    def test_all_parameterless_calculations(self, api, local_zarr_path, temp_dir):
        """Run all 9 parameter-free calculations in one call, all produce output."""
        calcs = [
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "evenness",
            "total_biomass",
            "total_biomass_comparison",
            "dominant_species",
            "rare_species",
            "common_species",
        ]
        results = api.calculate_metrics(
            local_zarr_path,
            calculations=calcs,
            output_dir=temp_dir / "all_calcs",
        )

        assert len(results) == len(calcs)
        for result in results:
            assert result.output_path.exists(), f"{result.name} output missing"
            assert result.output_path.stat().st_size > 0

    def test_invalid_calculation_name(self, api, local_zarr_path, temp_dir):
        """Unknown calculation name raises CalculationFailed."""
        with pytest.raises(CalculationFailed, match="Unknown calculations"):
            api.calculate_metrics(
                local_zarr_path,
                calculations=["nonexistent_calculation_xyz"],
                output_dir=temp_dir / "bad_calc",
            )


# ---------------------------------------------------------------------------
# 6. Calculate With Stats
# ---------------------------------------------------------------------------

class TestCalculateWithStats:
    """Tests for calculate_metrics_with_stats with bootstrap CIs (~3 min)."""

    def test_stats_default_calculations(self, local_zarr_path):
        """Default calculations return 4 StatisticalResult objects with valid fields."""
        from gridfia.core.analysis.statistical_analysis import StatisticalResult

        api = GridFIA(seed=42)
        results = api.calculate_metrics_with_stats(
            local_zarr_path,
            n_bootstrap=100,  # Fewer for speed in tests
        )

        assert len(results) == 4
        expected_names = {"species_richness", "shannon_diversity", "simpson_diversity", "total_biomass"}
        assert set(results.keys()) == expected_names

        for name, result in results.items():
            assert isinstance(result, StatisticalResult), f"{name} is not StatisticalResult"
            assert np.isfinite(result.value), f"{name} value is not finite"
            assert result.confidence_interval[0] <= result.value, (
                f"{name} CI lower > value"
            )
            assert result.confidence_interval[1] >= result.value, (
                f"{name} CI upper < value"
            )
            assert result.standard_error > 0, f"{name} SE should be positive"
            assert result.method == "bootstrap"

    def test_stats_custom_confidence(self, local_zarr_path):
        """99% CI is wider than 95% CI for species_richness."""
        api = GridFIA(seed=42)

        results_95 = api.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["species_richness"],
            n_bootstrap=100,
            confidence_level=0.95,
        )

        api2 = GridFIA(seed=42)
        results_99 = api2.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["species_richness"],
            n_bootstrap=100,
            confidence_level=0.99,
        )

        ci_95 = results_95["species_richness"].confidence_interval
        ci_99 = results_99["species_richness"].confidence_interval

        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 >= width_95, "99% CI should be at least as wide as 95% CI"

    def test_seed_reproducibility(self, local_zarr_path):
        """Same seed=42 produces identical value, CI, SE."""
        api1 = GridFIA(seed=42)
        results1 = api1.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["shannon_diversity"],
            n_bootstrap=100,
        )

        api2 = GridFIA(seed=42)
        results2 = api2.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["shannon_diversity"],
            n_bootstrap=100,
        )

        r1 = results1["shannon_diversity"]
        r2 = results2["shannon_diversity"]

        assert r1.value == r2.value, "Same seed should produce identical point estimate"
        assert r1.confidence_interval == r2.confidence_interval, "Same seed should produce identical CI"
        assert r1.standard_error == r2.standard_error, "Same seed should produce identical SE"

    def test_different_seeds_different_ci(self, local_zarr_path):
        """Different seeds yield same point estimate but different CI bounds."""
        api1 = GridFIA(seed=42)
        results1 = api1.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["shannon_diversity"],
            n_bootstrap=100,
        )

        api2 = GridFIA(seed=999)
        results2 = api2.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["shannon_diversity"],
            n_bootstrap=100,
        )

        r1 = results1["shannon_diversity"]
        r2 = results2["shannon_diversity"]

        # Point estimate is deterministic (no randomness), so same value
        assert r1.value == r2.value, "Point estimate should be identical regardless of seed"

        # CI bounds should differ because bootstrap resampling is different
        assert r1.confidence_interval != r2.confidence_interval, (
            "Different seeds should produce different CI bounds"
        )


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------

class TestVisualization:
    """Tests for create_maps with real Zarr data (~2 min)."""

    def test_species_map(self, api, local_zarr_path, temp_dir):
        """Single species PNG exists and is > 1KB."""
        maps = api.create_maps(
            local_zarr_path,
            map_type="species",
            species=["0316"],  # Red maple
            output_dir=temp_dir / "maps_species",
        )
        assert len(maps) == 1
        assert maps[0].exists()
        assert maps[0].stat().st_size > 1024, "Map file should be > 1KB"

    def test_diversity_maps(self, api, local_zarr_path, temp_dir):
        """Diversity maps create shannon_diversity.png and simpson_diversity.png."""
        output_dir = temp_dir / "maps_diversity"
        maps = api.create_maps(
            local_zarr_path,
            map_type="diversity",
            output_dir=output_dir,
        )
        assert len(maps) == 2

        filenames = {m.name for m in maps}
        assert "shannon_diversity.png" in filenames
        assert "simpson_diversity.png" in filenames

        for m in maps:
            assert m.exists()
            assert m.stat().st_size > 1024

    def test_richness_map(self, api, local_zarr_path, temp_dir):
        """Richness map creates species_richness.png."""
        output_dir = temp_dir / "maps_richness"
        maps = api.create_maps(
            local_zarr_path,
            map_type="richness",
            output_dir=output_dir,
        )
        assert len(maps) == 1
        assert maps[0].name == "species_richness.png"
        assert maps[0].exists()

    def test_comparison_map(self, api, local_zarr_path, temp_dir):
        """2-species comparison creates species_comparison.png."""
        output_dir = temp_dir / "maps_comparison"

        # Get available species codes from the store
        store = ZarrStore.from_path(local_zarr_path)
        codes = [c for c in store.species_codes if c != "0000"][:2]
        assert len(codes) >= 2, "Need at least 2 non-total species for comparison"

        maps = api.create_maps(
            local_zarr_path,
            map_type="comparison",
            species=codes,
            output_dir=output_dir,
        )
        assert len(maps) == 1
        assert maps[0].name == "species_comparison.png"
        assert maps[0].exists()

    def test_map_invalid_type(self, api, local_zarr_path, temp_dir):
        """Invalid map type raises CalculationFailed."""
        with pytest.raises(CalculationFailed, match="Unknown map type"):
            api.create_maps(
                local_zarr_path,
                map_type="nonexistent_type",
                output_dir=temp_dir / "maps_bad",
            )

    def test_comparison_too_few_species(self, api, local_zarr_path, temp_dir):
        """Comparison with 1 species raises SpeciesNotFound."""
        with pytest.raises(SpeciesNotFound, match="at least 2"):
            api.create_maps(
                local_zarr_path,
                map_type="comparison",
                species=["0316"],
                output_dir=temp_dir / "maps_few",
            )


# ---------------------------------------------------------------------------
# 8. Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end pipeline test combining all API operations (~2 min)."""

    def test_complete_pipeline(self, api, local_zarr_path, temp_dir):
        """validate -> calculate_metrics (3 calcs) -> stats -> create_maps (diversity)."""
        # Step 1: Validate
        info = api.validate_zarr(local_zarr_path)
        assert info["num_species"] >= 3

        # Step 2: Calculate metrics
        calc_results = api.calculate_metrics(
            local_zarr_path,
            calculations=["species_richness", "shannon_diversity", "total_biomass"],
            output_dir=temp_dir / "pipeline_calcs",
        )
        assert len(calc_results) == 3
        for result in calc_results:
            assert result.output_path.exists()

        # Step 3: Stats with bootstrap CIs
        stats_api = GridFIA(seed=42)
        stat_results = stats_api.calculate_metrics_with_stats(
            local_zarr_path,
            calculations=["species_richness", "shannon_diversity"],
            n_bootstrap=50,
        )
        assert len(stat_results) == 2
        for name, sr in stat_results.items():
            assert np.isfinite(sr.value)

        # Step 4: Visualization
        maps = api.create_maps(
            local_zarr_path,
            map_type="diversity",
            output_dir=temp_dir / "pipeline_maps",
        )
        assert len(maps) == 2
        for m in maps:
            assert m.exists()

    def test_download_create_zarr_roundtrip(self, api, temp_dir):
        """Download species → create_zarr → validate → ZarrStore read roundtrip."""
        # Download 2 species for a very small area
        bbox_wm = (-8763000, 4272000, -8761000, 4274000)
        download_dir = temp_dir / "roundtrip_dl"
        download_dir.mkdir()

        files = api.download_species(
            output_dir=download_dir,
            species_codes=["0316", "0131"],
            bbox=bbox_wm,
            crs="102100",
        )
        assert len(files) >= 1

        # Create Zarr
        zarr_path = temp_dir / "roundtrip.zarr"
        api.create_zarr(download_dir, zarr_path)
        assert zarr_path.exists()

        # Validate
        info = api.validate_zarr(zarr_path)
        assert info["num_species"] >= 2

        # Read via ZarrStore
        store = ZarrStore.from_path(zarr_path)
        assert store.shape[0] >= 2
        assert store.height > 0
        assert store.width > 0

        # Data should have non-zero values (forested area)
        total_biomass = store.biomass[0, :, :]
        assert np.nansum(total_biomass) > 0
