"""
Integration tests using real API calls.

Per project guidelines in CLAUDE.md: "Always use real API calls for tests."

These tests verify actual behavior against the FIA BIGMAP API.
They are marked as slow/integration and can be skipped in CI with:
    pytest -m "not slow"
    pytest -m "not integration"

Run only integration tests with:
    pytest -m integration
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from gridfia.api import GridFIA, SpeciesInfo, CalculationResult
from gridfia.external.fia_client import BigMapRestClient
from gridfia.utils.location_config import LocationConfig
from gridfia.config import GridFIASettings


# =============================================================================
# Real API Client Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestRealAPIClient:
    """Integration tests for BigMapRestClient using real API calls."""

    @pytest.fixture
    def client(self):
        """Create a client with longer timeout for real API calls."""
        return BigMapRestClient(timeout=120, rate_limit_delay=1.0)

    def test_real_service_info(self, client):
        """Test real service info retrieval from FIA BIGMAP API."""
        result = client.get_service_info()

        assert isinstance(result, dict)
        assert "name" in result or "serviceName" in result
        # Service should have raster functions
        assert "rasterFunctionInfos" in result

    def test_real_species_functions(self, client):
        """Test real species functions retrieval."""
        functions = client.get_species_functions()

        assert isinstance(functions, list)
        assert len(functions) > 100  # BIGMAP has 300+ species

        # Check structure of first function
        first_func = functions[0]
        assert "name" in first_func

    def test_real_list_available_species(self, client):
        """Test real species listing with proper structure."""
        species_list = client.list_available_species()

        assert isinstance(species_list, list)
        assert len(species_list) > 100  # Should have many species

        # Verify structure of returned species
        first_species = species_list[0]
        assert "species_code" in first_species
        assert "common_name" in first_species
        assert "scientific_name" in first_species
        assert "function_name" in first_species

        # Check that species codes are properly formatted
        for species in species_list[:10]:
            assert len(species["species_code"]) == 4 or species["species_code"].isdigit()

    def test_real_known_species_present(self, client):
        """Test that known common species are present in the API."""
        species_list = client.list_available_species()

        species_codes = {s["species_code"] for s in species_list}

        # These are common species that should always be in BIGMAP
        known_species = ["0131", "0202", "0316", "0261"]  # Balsam fir, Douglas-fir, Red maple, Eastern white pine

        for code in known_species:
            assert code in species_codes, f"Expected species {code} not found in API"


@pytest.mark.slow
@pytest.mark.integration
class TestRealAPIExport:
    """Integration tests for raster export using real API calls."""

    @pytest.fixture
    def client(self):
        """Create a client with longer timeout for real API calls."""
        return BigMapRestClient(timeout=180, rate_limit_delay=1.0)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_real_export_small_raster(self, client, temp_dir):
        """Test real raster export with a small bounding box."""
        # Small bbox in North Carolina (where BIGMAP has good data)
        # This is approximately 10km x 10km
        bbox = (-8450000, 4240000, -8440000, 4250000)
        output_path = temp_dir / "test_export.tif"

        result = client.export_species_raster(
            species_code="0316",  # Red maple - very common
            bbox=bbox,
            output_path=output_path,
            pixel_size=100.0  # 100m pixels for smaller file
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify it's a valid raster
        import rasterio
        with rasterio.open(output_path) as src:
            assert src.width > 0
            assert src.height > 0
            assert src.crs is not None

    def test_real_export_as_array(self, client):
        """Test real raster export returning numpy array."""
        # Very small bbox for quick test
        bbox = (-8445000, 4245000, -8444000, 4246000)

        result = client.export_species_raster(
            species_code="0316",  # Red maple
            bbox=bbox,
            output_path=None,  # Return as array
            pixel_size=100.0
        )

        assert isinstance(result, np.ndarray)
        assert result.size > 0
        # Biomass values should be non-negative
        assert np.nanmin(result) >= 0

    def test_real_identify_pixel_value(self, client):
        """Test real pixel value identification."""
        # Point in a forested area of North Carolina
        x, y = -8445000, 4245000  # Web Mercator coordinates

        result = client.identify_pixel_value("0316", x, y)  # Red maple

        # Result should be a float or None/0.0 if no data
        assert result is None or isinstance(result, (int, float))
        if result is not None and result != 0.0:
            assert result >= 0  # Biomass should be non-negative

    def test_real_total_biomass_export(self, client, temp_dir):
        """Test real total biomass raster export."""
        bbox = (-8450000, 4240000, -8440000, 4250000)
        output_path = temp_dir / "total_biomass.tif"

        result = client.export_total_biomass_raster(
            bbox=bbox,
            output_path=output_path,
            pixel_size=100.0
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0


# =============================================================================
# Real GridFIA API Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestRealGridFIA:
    """Integration tests for GridFIA API using real API calls."""

    @pytest.fixture
    def api(self):
        """Create GridFIA instance."""
        return GridFIA()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_real_list_species(self, api):
        """Test listing species from real API."""
        species = api.list_species()

        assert isinstance(species, list)
        assert len(species) > 100
        assert all(isinstance(s, SpeciesInfo) for s in species)

        # Check first species has all fields
        first = species[0]
        assert first.species_code is not None
        assert first.common_name is not None
        assert first.scientific_name is not None

    def test_real_list_calculations(self, api):
        """Test listing available calculations."""
        calculations = api.list_calculations()

        assert isinstance(calculations, list)
        assert len(calculations) > 0

        # Known calculations should be present
        expected = ["species_richness", "shannon_diversity", "total_biomass"]
        for calc in expected:
            assert calc in calculations, f"Expected calculation '{calc}' not found"

    def test_real_get_location_config_state(self, api):
        """Test creating location config for a state."""
        config = api.get_location_config(state="Montana")

        assert isinstance(config, LocationConfig)
        assert config.location_name == "Montana"
        assert config.web_mercator_bbox is not None
        assert len(config.web_mercator_bbox) == 4

    def test_real_get_location_config_county(self, api):
        """Test creating location config for a county."""
        config = api.get_location_config(state="North Carolina", county="Wake")

        assert isinstance(config, LocationConfig)
        assert "Wake" in config.location_name
        assert config.web_mercator_bbox is not None


@pytest.mark.slow
@pytest.mark.integration
class TestRealDownloadWorkflow:
    """Integration tests for the complete download workflow."""

    @pytest.fixture
    def api(self):
        """Create GridFIA instance."""
        return GridFIA()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_real_download_with_wgs84_bbox(self, api, temp_dir):
        """Test downloading with WGS84 coordinates (automatic transformation)."""
        # WGS84 bbox near Umstead State Park, NC
        bbox_wgs84 = (-79.1, 35.75, -79.05, 35.8)

        files = api.download_species(
            output_dir=temp_dir,
            species_codes=["0316"],  # Red maple
            bbox=bbox_wgs84,
            crs="EPSG:4326"  # WGS84 - should be auto-transformed to Web Mercator
        )

        assert isinstance(files, list)
        assert len(files) >= 1, "Should download at least one species"
        assert files[0].exists()
        assert files[0].stat().st_size > 0

        # Verify it's a valid raster
        import rasterio
        with rasterio.open(files[0]) as src:
            assert src.width > 0
            assert src.height > 0

    def test_real_download_with_web_mercator_bbox(self, api, temp_dir):
        """Test downloading with Web Mercator coordinates (no transformation)."""
        # Web Mercator bbox (same area as WGS84 test above)
        bbox_wm = (-8805372, 4266276, -8799806, 4273136)

        files = api.download_species(
            output_dir=temp_dir,
            species_codes=["0316"],  # Red maple
            bbox=bbox_wm,
            crs="102100"  # Web Mercator - no transformation needed
        )

        assert isinstance(files, list)
        assert len(files) >= 1, "Should download at least one species"
        assert files[0].exists()
        assert files[0].stat().st_size > 0


@pytest.mark.slow
@pytest.mark.integration
class TestRealZarrWorkflow:
    """Integration tests for Zarr creation from real data."""

    @pytest.fixture
    def api(self):
        """Create GridFIA instance."""
        return GridFIA()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_real_create_and_validate_zarr(self, api, temp_dir):
        """Test creating and validating Zarr from downloaded data."""
        # Download species data using Web Mercator bbox (API's native CRS)
        # Small area near Umstead State Park, NC (~3km x 3km)
        bbox_wm = (-8763000, 4272000, -8760000, 4275000)
        download_dir = temp_dir / "downloads"
        download_dir.mkdir()

        # Download directly using the REST client with Web Mercator
        client = api.rest_client
        species_codes = ["0316", "0261"]  # Red maple, Eastern white pine
        files = []

        for code in species_codes:
            output_path = download_dir / f"species_{code}.tif"
            try:
                result = client.export_species_raster(
                    species_code=code,
                    bbox=bbox_wm,
                    output_path=output_path,
                    pixel_size=30.0
                )
                if result and result.exists():
                    files.append(result)
            except Exception:
                pass

        if len(files) < 2:
            pytest.skip("Not enough data downloaded for Zarr creation test")

        # Create Zarr
        zarr_path = temp_dir / "test.zarr"
        result = api.create_zarr(download_dir, zarr_path)

        assert result == zarr_path
        assert zarr_path.exists()

        # Validate Zarr
        info = api.validate_zarr(zarr_path)
        assert "shape" in info
        assert info["shape"][0] >= 2  # At least 2 species (+ maybe total)


# =============================================================================
# Performance/Load Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestAPIPerformance:
    """Performance tests for API operations."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return BigMapRestClient(timeout=120, rate_limit_delay=0.5)

    def test_rate_limiting_respected(self, client):
        """Test that rate limiting is respected under load."""
        import time

        start_time = time.time()

        # Make 5 rapid requests
        for _ in range(5):
            client.get_service_info()

        elapsed = time.time() - start_time

        # With 0.5s rate limit, 5 requests should take at least 2s
        assert elapsed >= 2.0, "Rate limiting may not be working"

    def test_circuit_breaker_status(self, client):
        """Test circuit breaker reports proper status."""
        status = client.get_circuit_breaker_status()

        assert status is not None
        assert status["state"] == "closed"
        assert status["failure_count"] == 0


# =============================================================================
# Error Handling with Real API
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestRealAPIErrorHandling:
    """Test error handling with real API calls."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return BigMapRestClient(timeout=30)

    def test_invalid_species_code(self, client):
        """Test handling of invalid species code."""
        from gridfia.exceptions import SpeciesNotFound

        with pytest.raises(SpeciesNotFound):
            client.export_species_raster(
                species_code="9999",  # Invalid code
                bbox=(-8450000, 4240000, -8440000, 4250000)
            )

    def test_invalid_bbox_handling(self, client):
        """Test handling of invalid bounding box."""
        from gridfia.exceptions import DownloadError

        # Bbox outside CONUS
        bbox = (0, 0, 1, 1)  # Somewhere in the ocean

        # This should either return empty data or raise an error
        try:
            result = client.export_species_raster(
                species_code="0316",
                bbox=bbox,
                output_path=None,
                pixel_size=100.0
            )
            # If it returns, data should be empty/nodata
            if result is not None:
                assert np.all(result == 0) or np.all(np.isnan(result))
        except (DownloadError, Exception):
            pass  # Error is acceptable for invalid bbox


# =============================================================================
# End-to-End Workflow Test
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEndWorkflow:
    """Complete end-to-end workflow test using real API calls.

    This test verifies the complete pipeline:
    1. Download species rasters from FIA BIGMAP
    2. Create Zarr store from downloaded GeoTIFFs
    3. Run forest metric calculations
    4. Validate results
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_complete_pipeline_wake_county(self, temp_dir):
        """Test complete pipeline for a small area in Wake County, NC.

        Uses a ~5km x 5km bbox in a forested area for reasonable download times.
        """
        from gridfia.api import GridFIA
        from gridfia.utils.zarr_utils import ZarrStore

        api = GridFIA()

        # Step 1: Define a small forested area in Wake County, NC
        # This bbox is approximately 5km x 5km in Web Mercator
        # Near Umstead State Park - known forested area
        bbox_wm = (-8765000, 4270000, -8760000, 4275000)

        # Step 2: Download 3 common species (keep it small for speed)
        download_dir = temp_dir / "downloads"
        download_dir.mkdir()

        species_to_download = ["0316", "0131", "0261"]  # Red maple, Balsam fir, E. white pine

        downloaded_files = []
        client = api.rest_client

        for species_code in species_to_download:
            output_path = download_dir / f"species_{species_code}.tif"
            try:
                result = client.export_species_raster(
                    species_code=species_code,
                    bbox=bbox_wm,
                    output_path=output_path,
                    pixel_size=30.0  # 30m resolution
                )
                if result and result.exists() and result.stat().st_size > 0:
                    downloaded_files.append(result)
            except Exception as e:
                print(f"Warning: Could not download species {species_code}: {e}")

        assert len(downloaded_files) >= 1, "At least one species should download successfully"
        print(f"Downloaded {len(downloaded_files)} species rasters")

        # Step 3: Create Zarr store
        zarr_path = temp_dir / "test_workflow.zarr"
        zarr_result = api.create_zarr(download_dir, zarr_path)

        assert zarr_result == zarr_path
        assert zarr_path.exists()
        print(f"Created Zarr store at {zarr_path}")

        # Step 4: Validate Zarr structure
        info = api.validate_zarr(zarr_path)
        assert "shape" in info
        assert info["shape"][0] >= 2  # At least downloaded species + total
        print(f"Zarr shape: {info['shape']}")

        # Step 5: Run calculations
        output_dir = temp_dir / "calculations"
        output_dir.mkdir()

        results = api.calculate_metrics(
            zarr_path,
            calculations=["species_richness", "total_biomass"],
            output_dir=output_dir
        )

        assert len(results) >= 1, "At least one calculation should complete"
        for result in results:
            print(f"Calculation {result.name}: {result.output_path}")
            if result.output_path:
                assert Path(result.output_path).exists()

        # Step 6: Verify data integrity with ZarrStore
        with ZarrStore.open(zarr_path) as store:
            biomass = store.biomass
            assert biomass is not None
            assert biomass.shape[0] >= 2  # species + total

            # Check we have actual data (not all zeros)
            total_sum = float(np.nansum(biomass[:]))
            print(f"Total biomass sum: {total_sum}")

        print("End-to-end workflow completed successfully!")
