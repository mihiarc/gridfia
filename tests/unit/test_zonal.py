"""
Tests for gridfia.core.zonal — zonal statistics using exactextract.

Tests cover:
- calculate_zonal_stats (GeoTIFF-based)
- calculate_zonal_stats_from_zarr (Zarr-based with species name labels)
- _normalize_raster_input helper
- _sanitize_name helper
- CRS reprojection
- Edge cases (empty zones, missing columns, single vs multi raster)
"""

import tempfile
from pathlib import Path
from typing import Generator

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import zarr
import zarr.codecs
import zarr.storage
from rasterio.transform import from_bounds
from shapely.geometry import box

from gridfia.core.zonal import (
    DEFAULT_STATS,
    _normalize_raster_input,
    _sanitize_name,
    calculate_zonal_stats,
    calculate_zonal_stats_from_zarr,
)
from gridfia.utils.zarr_utils import (
    ZarrStore,
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


@pytest.fixture
def sample_geotiff(temp_dir: Path, raster_bounds) -> Path:
    """Create a small GeoTIFF raster for zonal stat tests."""
    path = temp_dir / "biomass.tif"
    height, width = 100, 100
    transform = from_bounds(*raster_bounds, width, height)

    rng = np.random.default_rng(42)
    data = (rng.random((height, width)) * 80).astype(np.float32)

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs="EPSG:5070",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def second_geotiff(temp_dir: Path, raster_bounds) -> Path:
    """Create a second raster for multi-raster tests."""
    path = temp_dir / "diversity.tif"
    height, width = 100, 100
    transform = from_bounds(*raster_bounds, width, height)

    rng = np.random.default_rng(99)
    data = (rng.random((height, width)) * 5).astype(np.float32)

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs="EPSG:5070",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path


@pytest.fixture
def zones_gdf(raster_bounds) -> gpd.GeoDataFrame:
    """Create polygon zones that overlap the test rasters, in matching CRS."""
    left, bottom, right, top = raster_bounds
    mid_x = (left + right) / 2
    mid_y = (bottom + top) / 2

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
    mid_y = (bottom + top) / 2

    # Create in raster CRS then reproject to 4326
    west_zone = box(left, bottom, mid_x, top)
    east_zone = box(mid_x, bottom, right, top)

    gdf = gpd.GeoDataFrame(
        {"name": ["west", "east"], "zone_id": [1, 2]},
        geometry=[west_zone, east_zone],
        crs="EPSG:5070",
    )
    return gdf.to_crs("EPSG:4326")


@pytest.fixture
def zarr_store_path(temp_dir: Path, raster_bounds) -> Path:
    """Create a complete ZarrStore with species data for zonal tests."""
    left, bottom, right, top = raster_bounds
    height, width = 100, 100
    transform = from_bounds(left, bottom, right, top, width, height)

    zarr_path = temp_dir / "forest.zarr"
    store = zarr.storage.LocalStore(str(zarr_path))
    root = zarr.open_group(store=store, mode="w")

    n_species = 4  # total + 3 species
    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3)
    biomass = root.create_array(
        "biomass",
        shape=(n_species, height, width),
        chunks=(1, 50, 50),
        dtype="f4",
        compressors=[codec],
        fill_value=0,
    )

    rng = np.random.default_rng(42)
    species_data = []
    for i in range(1, n_species):
        layer = (rng.random((height, width)) * 50).astype(np.float32)
        biomass[i, :, :] = layer
        species_data.append(layer)

    # Total biomass = sum of species
    biomass[0, :, :] = sum(species_data)

    # Species metadata as arrays (the way create_expandable_zarr works)
    codes_arr = root.create_array(
        "species_codes", shape=(n_species,), dtype="<U10", fill_value=""
    )
    names_arr = root.create_array(
        "species_names", shape=(n_species,), dtype="<U100", fill_value=""
    )
    codes_arr[0] = "0000"
    codes_arr[1] = "0202"
    codes_arr[2] = "0122"
    codes_arr[3] = "0746"
    names_arr[0] = "Total Biomass"
    names_arr[1] = "Douglas-fir"
    names_arr[2] = "Ponderosa Pine"
    names_arr[3] = "Quaking Aspen"

    root.attrs["crs"] = "EPSG:5070"
    root.attrs["transform"] = list(transform)[:6]
    root.attrs["bounds"] = [left, bottom, right, top]
    root.attrs["num_species"] = n_species

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
        with pytest.raises(FileNotFoundError):
            calculate_zonal_stats_from_zarr("/does/not/exist.zarr", zones_gdf)

    def test_empty_zones_raises(self, zarr_store_path):
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:5070")
        with pytest.raises(ValueError, match="empty"):
            calculate_zonal_stats_from_zarr(zarr_store_path, empty_gdf)

    def test_invalid_species_code_raises(self, zarr_store_path, zones_gdf):
        with pytest.raises(Exception, match="9999"):
            calculate_zonal_stats_from_zarr(
                zarr_store_path, zones_gdf, layers=["9999"]
            )

    def test_values_are_reasonable(self, zarr_store_path, zones_gdf):
        """Stat values should be within expected ranges for our test data."""
        result = calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean", "min", "max"]
        )
        # Test data is rng.random() * 50, so values in [0, 50]
        for name in ["douglas_fir", "ponderosa_pine", "quaking_aspen"]:
            assert (result[f"{name}_mean"] >= 0).all()
            assert (result[f"{name}_mean"] <= 50).all()
            assert (result[f"{name}_min"] >= 0).all()
            assert (result[f"{name}_max"] <= 50).all()

    def test_store_is_closed_after_call(self, zarr_store_path, zones_gdf):
        """ZarrStore should be closed even if no error occurs."""
        # This is an implicit test — if the store weren't closed, repeated
        # calls in the same process could leak resources. We verify by
        # calling twice successfully.
        calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
        calculate_zonal_stats_from_zarr(
            zarr_store_path, zones_gdf, stats=["mean"]
        )
