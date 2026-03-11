"""
Tests for gridfia.core.zonal — zonal statistics using exactextract.

Tests cover:
- calculate_zonal_stats (GeoTIFF-based)
- calculate_zonal_stats_from_zarr (Zarr-based with species name labels)
- _normalize_raster_input helper
- _sanitize_name helper
- _resolve_layer_labels collision detection
- _load_and_prepare_zones shared helper
- CRS reprojection
- Nodata handling (fill_value=0 excluded from stats)
- Edge cases (empty zones, missing columns, single vs multi raster)
- ZarrStore lifecycle (context manager cleanup)
"""

import tempfile
from pathlib import Path
from typing import Generator
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from gridfia.core.zonal import (
    DEFAULT_STATS,
    _load_and_prepare_zones,
    _normalize_raster_input,
    _resolve_layer_labels,
    _sanitize_name,
    calculate_zonal_stats,
    calculate_zonal_stats_from_zarr,
)
from gridfia.exceptions import SpeciesNotFound
from gridfia.utils.zarr_utils import (
    ZarrStore,
    append_species_to_zarr,
    create_expandable_zarr_from_base_raster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def raster_bounds():
    """Shared spatial extent in EPSG:5070 (Conus Albers)."""
    return (-2000000.0, 2500000.0, -1997000.0, 2503000.0)


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    bounds: tuple,
    crs: str = "EPSG:5070",
    nodata: float = None,
) -> Path:
    """Helper to write a GeoTIFF with given data and spatial metadata."""
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def sample_geotiff(temp_dir: Path, raster_bounds) -> Path:
    """Create a small GeoTIFF raster for zonal stat tests."""
    rng = np.random.default_rng(42)
    data = (rng.random((100, 100)) * 80).astype(np.float32)
    return _write_geotiff(temp_dir / "biomass.tif", data, raster_bounds)


@pytest.fixture
def second_geotiff(temp_dir: Path, raster_bounds) -> Path:
    """Create a second raster for multi-raster tests."""
    rng = np.random.default_rng(99)
    data = (rng.random((100, 100)) * 5).astype(np.float32)
    return _write_geotiff(temp_dir / "diversity.tif", data, raster_bounds)


@pytest.fixture
def zones_gdf(raster_bounds) -> gpd.GeoDataFrame:
    """Create polygon zones that overlap the test rasters, in matching CRS."""
    left, bottom, right, top = raster_bounds
    mid_x = (left + right) / 2

    west_zone = box(left, bottom, mid_x, top)
    east_zone = box(mid_x, bottom, right, top)

    return gpd.GeoDataFrame(
        {"name": ["west", "east"], "zone_id": [1, 2]},
        geometry=[west_zone, east_zone],
        crs="EPSG:5070",
    )


@pytest.fixture
def zones_gdf_4326(raster_bounds) -> gpd.GeoDataFrame:
    """Zones in EPSG:4326 to test CRS reprojection."""
    left, bottom, right, top = raster_bounds
    mid_x = (left + right) / 2

    west_zone = box(left, bottom, mid_x, top)
    east_zone = box(mid_x, bottom, right, top)

    gdf = gpd.GeoDataFrame(
        {"name": ["west", "east"], "zone_id": [1, 2]},
        geometry=[west_zone, east_zone],
        crs="EPSG:5070",
    )
    return gdf.to_crs("EPSG:4326")


@pytest.fixture
def zones_geojson(temp_dir: Path, zones_gdf) -> Path:
    """Save zones to a GeoJSON file for file-based loading tests."""
    path = temp_dir / "zones.geojson"
    zones_gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def zarr_store_path(temp_dir: Path, raster_bounds) -> Path:
    """Create a Zarr store via the production API (create + append).

    Uses create_expandable_zarr_from_base_raster and append_species_to_zarr
    to ensure round-trip compatibility with ZarrStore.
    """
    left, bottom, right, top = raster_bounds
    height, width = 100, 100

    rng = np.random.default_rng(42)

    # Create a base raster (total biomass) as a GeoTIFF
    total_data = (rng.random((height, width)) * 150).astype(np.float32)
    base_raster = _write_geotiff(
        temp_dir / "total_biomass.tif", total_data, raster_bounds
    )

    # Create species rasters
    species = [
        ("0202", "Douglas-fir"),
        ("0122", "Ponderosa Pine"),
        ("0746", "Quaking Aspen"),
    ]
    species_rasters = []
    for code, name in species:
        data = (rng.random((height, width)) * 50).astype(np.float32)
        path = _write_geotiff(temp_dir / f"{code}.tif", data, raster_bounds)
        species_rasters.append((path, code, name))

    # Create Zarr store using production API
    zarr_path = temp_dir / "forest.zarr"
    create_expandable_zarr_from_base_raster(base_raster, zarr_path)

    for raster_path, code, name in species_rasters:
        append_species_to_zarr(zarr_path, raster_path, code, name)

    return zarr_path


@pytest.fixture
def zarr_store_with_nodata(temp_dir: Path, raster_bounds) -> Path:
    """Zarr store where some pixels are zero (nodata) to test exclusion."""
    height, width = 100, 100
    rng = np.random.default_rng(42)

    # Create total biomass with zeros in top-left quadrant
    total_data = (rng.random((height, width)) * 100).astype(np.float32)
    total_data[:50, :50] = 0.0  # non-forested quadrant
    base_raster = _write_geotiff(
        temp_dir / "total_nodata.tif", total_data, raster_bounds
    )

    # Species with same zero pattern
    species_data = (rng.random((height, width)) * 40).astype(np.float32)
    species_data[:50, :50] = 0.0
    species_raster = _write_geotiff(
        temp_dir / "0202_nodata.tif", species_data, raster_bounds
    )

    zarr_path = temp_dir / "nodata_test.zarr"
    create_expandable_zarr_from_base_raster(base_raster, zarr_path)
    append_species_to_zarr(zarr_path, species_raster, "0202", "Douglas-fir")

    return zarr_path


# ---------------------------------------------------------------------------
# _sanitize_name tests
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_spaces_to_underscores(self):
        assert _sanitize_name("Loblolly Pine") == "loblolly_pine"

    def test_hyphens_to_underscores(self):
        assert _sanitize_name("Douglas-fir") == "douglas_fir"

    def test_mixed(self):
        assert _sanitize_name("Red-Cedar Pine") == "red_cedar_pine"

    def test_already_clean(self):
        assert _sanitize_name("oak") == "oak"

    def test_single_word(self):
        assert _sanitize_name("Total") == "total"


# ---------------------------------------------------------------------------
# _normalize_raster_input tests
# ---------------------------------------------------------------------------


class TestNormalizeRasterInput:
    def test_single_path_string(self):
        result = _normalize_raster_input("/data/richness.tif")
        assert result == {"richness": Path("/data/richness.tif")}

    def test_single_path_object(self):
        result = _normalize_raster_input(Path("/data/richness.tif"))
        assert result == {"richness": Path("/data/richness.tif")}

    def test_list_of_paths(self):
        result = _normalize_raster_input(["/data/a.tif", "/data/b.tif"])
        assert "a" in result
        assert "b" in result
        assert result["a"] == Path("/data/a.tif")

    def test_dict_passthrough(self):
        result = _normalize_raster_input({"rich": "/data/r.tif", "div": "/data/d.tif"})
        assert result == {"rich": Path("/data/r.tif"), "div": Path("/data/d.tif")}

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported raster input type"):
            _normalize_raster_input(42)


# ---------------------------------------------------------------------------
# _resolve_layer_labels tests
# ---------------------------------------------------------------------------


class TestResolveLayerLabels:
    def test_basic_labeling(self):
        labels, mapping = _resolve_layer_labels(
            [1, 2], ["0202", "0122"], ["Total", "Douglas-fir", "Ponderosa Pine"]
        )
        assert labels == ["douglas_fir", "ponderosa_pine"]
        assert mapping == {"0202": "Douglas-fir", "0122": "Ponderosa Pine"}

    def test_collision_disambiguates_with_code(self):
        """Two species that sanitize to the same label get code suffixes."""
        labels, mapping = _resolve_layer_labels(
            [1, 2],
            ["0001", "0002"],
            ["Total", "Red Cedar", "Red-Cedar"],
        )
        # Both would be "red_cedar", so they get disambiguated
        assert labels == ["red_cedar_0001", "red_cedar_0002"]

    def test_fallback_to_code_when_name_missing(self):
        labels, mapping = _resolve_layer_labels(
            [1, 2], ["0202", "0122"], ["Total"]  # only index 0 has a name
        )
        assert labels == ["0202", "0122"]
        assert mapping["0202"] == "0202"

    def test_no_collision_for_unique_names(self):
        labels, _ = _resolve_layer_labels(
            [1, 2, 3],
            ["0202", "0122", "0746"],
            ["Total", "Douglas-fir", "Ponderosa Pine", "Quaking Aspen"],
        )
        assert labels == ["douglas_fir", "ponderosa_pine", "quaking_aspen"]


# ---------------------------------------------------------------------------
# _load_and_prepare_zones tests
# ---------------------------------------------------------------------------


class TestLoadAndPrepareZones:
    def test_geodataframe_input(self, zones_gdf):
        from rasterio.crs import CRS

        gdf, cols = _load_and_prepare_zones(zones_gdf, CRS.from_epsg(5070))
        assert len(gdf) == 2
        assert "name" in cols
        assert "zone_id" in cols

    def test_include_cols_filters(self, zones_gdf):
        from rasterio.crs import CRS

        gdf, cols = _load_and_prepare_zones(
            zones_gdf, CRS.from_epsg(5070), include_cols=["name"]
        )
        assert cols == ["name"]

    def test_missing_include_cols_warns(self, zones_gdf):
        from rasterio.crs import CRS

        gdf, cols = _load_and_prepare_zones(
            zones_gdf,
            CRS.from_epsg(5070),
            include_cols=["name", "nonexistent"],
        )
        assert cols == ["name"]

    def test_empty_zones_raises(self):
        from rasterio.crs import CRS

        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:5070")
        with pytest.raises(ValueError, match="empty"):
            _load_and_prepare_zones(empty_gdf, CRS.from_epsg(5070))

    def test_crs_reprojection(self, zones_gdf_4326):
        from rasterio.crs import CRS

        target = CRS.from_epsg(5070)
        gdf, _ = _load_and_prepare_zones(zones_gdf_4326, target)
        assert gdf.crs.to_epsg() == 5070

    def test_file_based_zones(self, zones_geojson):
        from rasterio.crs import CRS

        gdf, cols = _load_and_prepare_zones(zones_geojson, CRS.from_epsg(5070))
        assert len(gdf) == 2
        assert "name" in cols


# ---------------------------------------------------------------------------
# calculate_zonal_stats (GeoTIFF) tests
# ---------------------------------------------------------------------------


class TestCalculateZonalStats:
    def test_single_raster_default_stats(self, sample_geotiff, zones_gdf):
        result = calculate_zonal_stats(sample_geotiff, zones_gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2  # two zones

        # Default stats should all be present (no raster prefix for single)
        for stat in DEFAULT_STATS:
            assert stat in result.columns, f"Missing column: {stat}"

        # Values should be reasonable
        assert result["mean"].notna().all()
        assert (result["count"] > 0).all()

    def test_single_raster_custom_stats(self, sample_geotiff, zones_gdf):
        result = calculate_zonal_stats(
            sample_geotiff, zones_gdf, stats=["mean", "max"]
        )
        assert "mean" in result.columns
        assert "max" in result.columns
        assert "sum" not in result.columns

    def test_multi_raster_prefixed_columns(
        self, sample_geotiff, second_geotiff, zones_gdf
    ):
        rasters = {"biomass": sample_geotiff, "diversity": second_geotiff}
        result = calculate_zonal_stats(rasters, zones_gdf, stats=["mean", "sum"])

        assert "biomass_mean" in result.columns
        assert "biomass_sum" in result.columns
        assert "diversity_mean" in result.columns
        assert "diversity_sum" in result.columns

    def test_list_of_rasters(self, sample_geotiff, second_geotiff, zones_gdf):
        result = calculate_zonal_stats(
            [sample_geotiff, second_geotiff], zones_gdf, stats=["mean"]
        )
        # Column names derived from filenames
        assert "biomass_mean" in result.columns
        assert "diversity_mean" in result.columns

    def test_include_cols(self, sample_geotiff, zones_gdf):
        result = calculate_zonal_stats(
            sample_geotiff, zones_gdf, stats=["mean"], include_cols=["name"]
        )
        assert "name" in result.columns
        assert "zone_id" not in result.columns

    def test_output_csv(self, sample_geotiff, zones_gdf, temp_dir):
        csv_path = temp_dir / "output" / "zonal.csv"
        result = calculate_zonal_stats(
            sample_geotiff, zones_gdf, stats=["mean"], output_csv=csv_path
        )
        assert csv_path.exists()

        import pandas as pd

        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == 2
        assert "mean" in csv_df.columns
        assert "geometry" not in csv_df.columns

    def test_preserves_geometry(self, sample_geotiff, zones_gdf):
        result = calculate_zonal_stats(sample_geotiff, zones_gdf, stats=["mean"])
        assert "geometry" in result.columns
        assert result.crs is not None

    def test_crs_reprojection(self, sample_geotiff, zones_gdf_4326):
        """Zones in EPSG:4326 should be auto-reprojected to match the raster."""
        result = calculate_zonal_stats(
            sample_geotiff, zones_gdf_4326, stats=["mean", "count"]
        )
        assert len(result) == 2
        assert (result["count"] > 0).all()

    def test_zones_from_file(self, sample_geotiff, zones_geojson):
        """Zones loaded from a GeoJSON file should work identically."""
        result = calculate_zonal_stats(
            sample_geotiff, zones_geojson, stats=["mean", "count"]
        )
        assert len(result) == 2
        assert (result["count"] > 0).all()

    def test_missing_raster_raises(self, zones_gdf):
        with pytest.raises(FileNotFoundError, match="Raster file not found"):
            calculate_zonal_stats("/does/not/exist.tif", zones_gdf)

    def test_empty_zones_raises(self, sample_geotiff):
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:5070")
        with pytest.raises(ValueError, match="empty"):
            calculate_zonal_stats(sample_geotiff, empty_gdf)


# ---------------------------------------------------------------------------
# calculate_zonal_stats_from_zarr tests
# ---------------------------------------------------------------------------


class TestCalculateZonalStatsFromZarr:
    def test_default_stats_all_species(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(zarr_store_path, zones_gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2

        # Should skip index 0 (total) and process 3 species
        # Columns named with sanitized species names
        assert "douglas_fir_mean" in result.columns
        assert "ponderosa_pine_mean" in result.columns
        assert "quaking_aspen_mean" in result.columns

        # All default stats present for each species
        for name in ["douglas_fir", "ponderosa_pine", "quaking_aspen"]:
            for stat in DEFAULT_STATS:
                col = f"{name}_{stat}"
                assert col in result.columns, f"Missing column: {col}"

    def test_human_readable_column_names(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
        # Should NOT have code-based columns
        assert "0202_mean" not in result.columns
        # Should have name-based columns
        assert "douglas_fir_mean" in result.columns

    def test_species_code_to_name_attrs(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
        mapping = result.attrs["species_code_to_name"]
        assert mapping["0202"] == "Douglas-fir"
        assert mapping["0122"] == "Ponderosa Pine"
        assert mapping["0746"] == "Quaking Aspen"

    def test_specific_layers(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, layers=["0202"], stats=["mean", "sum"]
        )
        assert "douglas_fir_mean" in result.columns
        assert "douglas_fir_sum" in result.columns
        # Other species should NOT be present
        assert "ponderosa_pine_mean" not in result.columns

    def test_custom_stats(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean", "max"]
        )
        assert "douglas_fir_mean" in result.columns
        assert "douglas_fir_max" in result.columns
        assert "douglas_fir_sum" not in result.columns

    def test_include_cols(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"], include_cols=["name"]
        )
        assert "name" in result.columns
        assert "zone_id" not in result.columns

    def test_output_csv(self, zarr_store_path, zones_gdf, temp_dir):
        csv_path = temp_dir / "zarr_zonal.csv"
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"], output_csv=csv_path
        )
        assert csv_path.exists()

        import pandas as pd

        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == 2
        assert "douglas_fir_mean" in csv_df.columns

    def test_crs_reprojection(self, zarr_store_path, zones_gdf_4326):
        """Zones in EPSG:4326 should be auto-reprojected to match Zarr CRS."""
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf_4326, stats=["mean", "count"]
        )
        assert len(result) == 2
        for name in ["douglas_fir", "ponderosa_pine", "quaking_aspen"]:
            assert (result[f"{name}_count"] > 0).all()

    def test_preserves_geometry(self, zarr_store_path, zones_gdf):
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
        assert "geometry" in result.columns
        assert result.crs is not None

    def test_nonexistent_zarr_raises(self, zones_gdf):
        """Core function should raise FileNotFoundError explicitly."""
        with pytest.raises(FileNotFoundError, match="Zarr store not found"):
            calculate_zonal_stats_from_zarr("/does/not/exist.zarr", zones_gdf)

    def test_empty_zones_raises(self, zarr_store_path):
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:5070")
        with pytest.raises(ValueError, match="empty"):
            calculate_zonal_stats_from_zarr(zarr_store_path, empty_gdf)

    def test_invalid_species_code_raises(self, zarr_store_path, zones_gdf):
        with pytest.raises(SpeciesNotFound):
            calculate_zonal_stats_from_zarr(
                zarr_store_path, zones_gdf, layers=["9999"]
            )

    def test_values_are_reasonable(self, zarr_store_path, zones_gdf):
        """Stat values should be within expected ranges for our test data."""
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean", "min", "max"], nodata=None
        )
        # Test data is rng.random() * 50, so values in [0, 50]
        for name in ["douglas_fir", "ponderosa_pine", "quaking_aspen"]:
            assert (result[f"{name}_mean"] >= 0).all()
            assert (result[f"{name}_mean"] <= 50).all()
            assert (result[f"{name}_min"] >= 0).all()
            assert (result[f"{name}_max"] <= 50).all()

    def test_store_closed_after_call(self, zarr_store_path, zones_gdf):
        """ZarrStore should be properly closed after the function returns.

        Verified by calling twice successfully — if resources leaked, the
        second call could fail. Also verifies via the context manager pattern
        used in the implementation (ZarrStore.open).
        """
        result1 = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
        result2 = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
        # Both calls should produce identical results
        np.testing.assert_allclose(
            result1["douglas_fir_mean"].values,
            result2["douglas_fir_mean"].values,
        )

    def test_zones_from_file(self, zarr_store_path, zones_geojson):
        """Zones loaded from a GeoJSON file should work."""
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_geojson, stats=["mean"]
        )
        assert len(result) == 2
        assert "douglas_fir_mean" in result.columns


# ---------------------------------------------------------------------------
# Nodata handling tests
# ---------------------------------------------------------------------------


class TestNodataHandling:
    def test_nodata_zero_excluded_from_mean(
        self, zarr_store_with_nodata, raster_bounds
    ):
        """With nodata=0.0 (default), zero pixels should not drag down the mean."""
        left, bottom, right, top = raster_bounds

        # Zone covering the entire raster
        full_zone = gpd.GeoDataFrame(
            {"name": ["full"]},
            geometry=[box(left, bottom, right, top)],
            crs="EPSG:5070",
        )

        result_with_nodata = calculate_zonal_stats_from_zarr(
            zarr_store_with_nodata, full_zone, stats=["mean", "count"], nodata=0.0
        )
        result_without_nodata = calculate_zonal_stats_from_zarr(
            zarr_store_with_nodata, full_zone, stats=["mean", "count"], nodata=None
        )

        # With nodata=0, count should be ~75% of total pixels (top-left quadrant is zero)
        count_with = result_with_nodata["douglas_fir_count"].iloc[0]
        count_without = result_without_nodata["douglas_fir_count"].iloc[0]
        assert count_with < count_without

        # Mean should be higher when zeros are excluded
        mean_with = result_with_nodata["douglas_fir_mean"].iloc[0]
        mean_without = result_without_nodata["douglas_fir_mean"].iloc[0]
        assert mean_with > mean_without

    def test_nodata_none_includes_zeros(self, zarr_store_with_nodata, raster_bounds):
        """With nodata=None, all pixels (including zeros) are included."""
        left, bottom, right, top = raster_bounds
        full_zone = gpd.GeoDataFrame(
            {"name": ["full"]},
            geometry=[box(left, bottom, right, top)],
            crs="EPSG:5070",
        )

        result = calculate_zonal_stats_from_zarr(
            zarr_store_with_nodata, full_zone, stats=["count"], nodata=None
        )
        # All 100*100 = 10000 pixels should be counted
        assert result["douglas_fir_count"].iloc[0] == 10000


# ---------------------------------------------------------------------------
# Batching performance tests
# ---------------------------------------------------------------------------


class TestBatching:
    def test_results_consistent_across_batch_boundaries(
        self, zarr_store_path, zones_gdf
    ):
        """Results should be identical regardless of internal batch size."""
        from gridfia.core import zonal

        # Run with batch size 1 (no batching benefit)
        original_batch_size = zonal._BATCH_SIZE
        try:
            zonal._BATCH_SIZE = 1
            result_unbatched = calculate_zonal_stats_from_zarr(
                zarr_store_path, zones_gdf, stats=["mean", "sum"]
            )

            zonal._BATCH_SIZE = 50
            result_batched = calculate_zonal_stats_from_zarr(
                zarr_store_path, zones_gdf, stats=["mean", "sum"]
            )
        finally:
            zonal._BATCH_SIZE = original_batch_size

        # Results must match exactly
        for col in result_batched.columns:
            if col == "geometry":
                continue
            if result_batched[col].dtype == np.float64:
                np.testing.assert_allclose(
                    result_batched[col].values,
                    result_unbatched[col].values,
                    rtol=1e-5,
                    err_msg=f"Mismatch in column {col}",
                )
            else:
                assert list(result_batched[col]) == list(result_unbatched[col])
