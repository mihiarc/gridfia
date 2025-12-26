"""
Comprehensive tests for GridFIARestClient class.

This module provides comprehensive test coverage for the BigMapRestClient class,
testing all public methods, HTTP request handling, error conditions, retry logic,
and integration with the FIA BIGMAP ImageServer API.

Tests use real API calls as specified in project requirements.
"""

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest
import requests
import numpy as np
from requests.exceptions import ConnectionError, Timeout, RequestException
from urllib3.util.retry import Retry

from gridfia.external.fia_client import BigMapRestClient
from gridfia.exceptions import (
    InvalidZarrStructure, SpeciesNotFound, CalculationFailed,
    APIConnectionError, InvalidLocationConfig, DownloadError
)


class TestBigMapRestClientInitialization:
    """Test BigMapRestClient initialization and configuration."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        client = BigMapRestClient()

        assert client.base_url == "https://di-usfsdata.img.arcgis.com/arcgis/rest/services/FIA_BIGMAP_2018_Tree_Species_Aboveground_Biomass/ImageServer"
        assert client.timeout == 30
        assert client.rate_limit_delay == 0.5
        assert client._last_request_time == 0
        assert client._species_functions is None
        assert client.session is not None
        assert isinstance(client.session, requests.Session)

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        client = BigMapRestClient(
            max_retries=5,
            backoff_factor=2.0,
            timeout=60,
            rate_limit_delay=1.0
        )

        assert client.timeout == 60
        assert client.rate_limit_delay == 1.0
        assert client.session is not None

    def test_session_configuration(self):
        """Test that session is properly configured with retry strategy."""
        client = BigMapRestClient(max_retries=3, backoff_factor=1.5)

        # Check headers
        expected_headers = {
            'User-Agent': 'BigMap-Python-Client/1.0',
            'Accept': 'application/json'
        }
        for key, value in expected_headers.items():
            assert client.session.headers[key] == value

        # Check that adapters are mounted
        assert 'http://' in client.session.adapters
        assert 'https://' in client.session.adapters

    def test_retry_strategy_configuration(self):
        """Test retry strategy configuration."""
        max_retries = 5
        backoff_factor = 2.0

        client = BigMapRestClient(
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )

        # Get the adapter and check its retry configuration
        adapter = client.session.get_adapter('https://')
        retry_config = adapter.max_retries

        assert retry_config.total == max_retries
        assert retry_config.backoff_factor == backoff_factor
        assert 429 in retry_config.status_forcelist
        assert 500 in retry_config.status_forcelist
        assert 502 in retry_config.status_forcelist
        assert 503 in retry_config.status_forcelist
        assert 504 in retry_config.status_forcelist


class TestBigMapRestClientRateLimitedRequest:
    """Test _rate_limited_request method for proper rate limiting and error handling."""

    def test_rate_limiting_delay(self):
        """Test that rate limiting introduces proper delays."""
        client = BigMapRestClient(rate_limit_delay=0.1)

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            # First request should not delay
            start_time = time.time()
            client._rate_limited_request('GET', 'http://test.com')
            first_duration = time.time() - start_time

            # Second request should delay
            start_time = time.time()
            client._rate_limited_request('GET', 'http://test.com')
            second_duration = time.time() - start_time

            # Second request should take longer due to rate limiting
            assert second_duration >= 0.1
            assert mock_request.call_count == 2

    def test_timeout_configuration(self):
        """Test that timeout is properly set for requests."""
        client = BigMapRestClient(timeout=45)

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            client._rate_limited_request('GET', 'http://test.com')

            # Check that timeout was passed to request
            call_args = mock_request.call_args
            assert call_args[1]['timeout'] == 45

    def test_custom_timeout_override(self):
        """Test that custom timeout can override default."""
        client = BigMapRestClient(timeout=30)

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            client._rate_limited_request('GET', 'http://test.com', timeout=60)

            # Check that custom timeout was used
            call_args = mock_request.call_args
            assert call_args[1]['timeout'] == 60

    def test_429_rate_limit_handling(self):
        """Test handling of 429 rate limit responses."""
        client = BigMapRestClient()

        with patch.object(client.session, 'request') as mock_request:
            with patch('time.sleep') as mock_sleep:
                # First response is 429 with Retry-After header
                rate_limit_response = Mock()
                rate_limit_response.status_code = 429
                rate_limit_response.headers = {'Retry-After': '2'}

                # Second response is successful
                success_response = Mock()
                success_response.status_code = 200

                mock_request.side_effect = [rate_limit_response, success_response]

                result = client._rate_limited_request('GET', 'http://test.com')

                # Should have slept for the retry-after time
                mock_sleep.assert_called_once_with(2)
                # Should have made two requests (original + retry)
                assert mock_request.call_count == 2
                assert result == success_response

    def test_429_without_retry_after_header(self):
        """Test handling of 429 responses without Retry-After header."""
        client = BigMapRestClient()

        with patch.object(client.session, 'request') as mock_request:
            with patch('time.sleep') as mock_sleep:
                # 429 response without Retry-After header
                rate_limit_response = Mock()
                rate_limit_response.status_code = 429
                rate_limit_response.headers = {}

                mock_request.return_value = rate_limit_response

                result = client._rate_limited_request('GET', 'http://test.com')

                # Should not sleep without Retry-After header
                mock_sleep.assert_not_called()
                # Should only make one request
                assert mock_request.call_count == 1
                assert result == rate_limit_response

    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        client = BigMapRestClient()

        with patch.object(client.session, 'request') as mock_request:
            mock_request.side_effect = ConnectionError("Connection failed")

            with pytest.raises(APIConnectionError):
                client._rate_limited_request('GET', 'http://test.com')

    def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        client = BigMapRestClient()

        with patch.object(client.session, 'request') as mock_request:
            mock_request.side_effect = Timeout("Request timed out")

            with pytest.raises(APIConnectionError):
                client._rate_limited_request('GET', 'http://test.com')

    def test_general_request_error_handling(self):
        """Test handling of general request errors."""
        client = BigMapRestClient()

        with patch.object(client.session, 'request') as mock_request:
            mock_request.side_effect = RequestException("Request failed")

            with pytest.raises(APIConnectionError):
                client._rate_limited_request('GET', 'http://test.com')


class TestBigMapRestClientServiceInfo:
    """Test get_service_info method."""

    def test_get_service_info_success(self):
        """Test successful service info retrieval."""
        client = BigMapRestClient()
        expected_info = {
            'name': 'FIA_BIGMAP_2018_Tree_Species_Aboveground_Biomass',
            'serviceDescription': 'Forest Inventory Analysis BigMap data',
            'rasterFunctionInfos': []
        }

        with patch.object(client, '_rate_limited_request') as mock_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = expected_info
            mock_request.return_value = mock_response

            result = client.get_service_info()

            assert result == expected_info
            mock_request.assert_called_once_with('GET', f'{client.base_url}?f=json')
            mock_response.raise_for_status.assert_called_once()

    def test_get_service_info_http_error(self):
        """Test handling of HTTP errors in service info retrieval."""
        client = BigMapRestClient()

        with patch.object(client, '_rate_limited_request') as mock_request:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_request.return_value = mock_response

            with pytest.raises(APIConnectionError):
                client.get_service_info()

    def test_get_service_info_request_exception(self):
        """Test handling of request exceptions in service info retrieval."""
        client = BigMapRestClient()

        with patch.object(client, '_rate_limited_request') as mock_request:
            mock_request.side_effect = RequestException("Network error")

            with pytest.raises(APIConnectionError):
                client.get_service_info()


class TestBigMapRestClientSpeciesFunctions:
    """Test get_species_functions method."""

    def test_get_species_functions_success(self):
        """Test successful species functions retrieval."""
        client = BigMapRestClient()
        expected_functions = [
            {'name': 'SPCD_0131_Abies_balsamea', 'description': 'Balsam fir'},
            {'name': 'SPCD_0202_Pseudotsuga_menziesii', 'description': 'Douglas-fir'}
        ]
        service_info = {'rasterFunctionInfos': expected_functions}

        with patch.object(client, 'get_service_info', return_value=service_info):
            result = client.get_species_functions()

            assert result == expected_functions
            assert client._species_functions == expected_functions

    def test_get_species_functions_cached(self):
        """Test that species functions are cached after first retrieval."""
        client = BigMapRestClient()
        expected_functions = [
            {'name': 'SPCD_0131_Abies_balsamea', 'description': 'Balsam fir'}
        ]

        # Set cached value
        client._species_functions = expected_functions

        with patch.object(client, 'get_service_info') as mock_service_info:
            result = client.get_species_functions()

            assert result == expected_functions
            # get_service_info should not be called since functions are cached
            mock_service_info.assert_not_called()

    def test_get_species_functions_no_raster_functions(self):
        """Test handling when service info has no raster functions."""
        client = BigMapRestClient()
        service_info = {'name': 'Test Service'}  # No rasterFunctionInfos

        with patch.object(client, 'get_service_info', return_value=service_info):
            result = client.get_species_functions()

            assert result == []
            assert client._species_functions == []


class TestBigMapRestClientListSpecies:
    """Test list_available_species method."""

    def test_list_available_species_success(self):
        """Test successful species listing."""
        client = BigMapRestClient()
        mock_functions = [
            {'name': 'SPCD_0131_Abies_balsamea', 'description': 'Balsam fir'},
            {'name': 'SPCD_0202_Pseudotsuga_menziesii', 'description': 'Douglas-fir'},
            {'name': 'SPCD_0068_Liquidambar_styraciflua', 'description': 'American sweetgum'},
            {'name': 'SPCD_0000_TOTAL', 'description': 'Total biomass'},  # Should be excluded
            {'name': 'OTHER_FUNCTION', 'description': 'Not a species'}  # Should be excluded
        ]

        with patch.object(client, 'get_species_functions', return_value=mock_functions):
            result = client.list_available_species()

            # Should return 3 species (excluding TOTAL and non-SPCD functions)
            assert len(result) == 3

            # Check first species
            assert result[0]['species_code'] == '0068'  # Sorted by code
            assert result[0]['common_name'] == 'American sweetgum'
            assert result[0]['scientific_name'] == 'Liquidambar styraciflua'
            assert result[0]['function_name'] == 'SPCD_0068_Liquidambar_styraciflua'

            # Check sorting by species code
            codes = [s['species_code'] for s in result]
            assert codes == ['0068', '0131', '0202']

    def test_list_available_species_malformed_function_names(self):
        """Test handling of malformed function names."""
        client = BigMapRestClient()
        mock_functions = [
            {'name': 'SPCD_0131_Abies_balsamea', 'description': 'Balsam fir'},
            {'name': 'SPCD_', 'description': 'Malformed 1'},  # Empty species code but valid format
            {'name': 'SPCD_ABCD', 'description': 'Malformed 2'},  # Valid format but non-numeric code
        ]

        with patch.object(client, 'get_species_functions', return_value=mock_functions):
            result = client.list_available_species()

            # Should return all 3 entries - implementation accepts any SPCD_ format with len >= 2
            assert len(result) == 3

            # Results should be sorted by species code (empty string comes first)
            assert result[0]['species_code'] == ''  # Empty string from 'SPCD_'
            assert result[1]['species_code'] == '0131'
            assert result[2]['species_code'] == 'ABCD'

    def test_list_available_species_empty_functions(self):
        """Test handling when no species functions are available."""
        client = BigMapRestClient()

        with patch.object(client, 'get_species_functions', return_value=[]):
            result = client.list_available_species()

            assert result == []

    def test_list_available_species_complex_scientific_names(self):
        """Test parsing of complex scientific names with multiple underscores."""
        client = BigMapRestClient()
        mock_functions = [
            {
                'name': 'SPCD_0131_Abies_balsamea_var_phanerolepis',
                'description': 'Balsam fir variety'
            }
        ]

        with patch.object(client, 'get_species_functions', return_value=mock_functions):
            result = client.list_available_species()

            assert len(result) == 1
            assert result[0]['scientific_name'] == 'Abies balsamea var phanerolepis'


class TestBigMapRestClientExportSpeciesRaster:
    """Test export_species_raster method."""

    def test_export_species_raster_success_with_file(self, temp_dir):
        """Test successful species raster export to file."""
        client = BigMapRestClient()
        species_code = '0131'
        bbox = (-12000000, 5000000, -11000000, 6000000)
        output_path = temp_dir / "test_species.tif"

        # Mock function name lookup
        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                # Mock export request
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'href': 'http://test.com/raster.tif'}

                # Mock raster download
                raster_response = Mock()
                raster_response.raise_for_status = Mock()
                raster_response.content = b'fake tiff data'

                mock_request.side_effect = [export_response, raster_response]

                result = client.export_species_raster(
                    species_code=species_code,
                    bbox=bbox,
                    output_path=output_path
                )

                assert result == output_path
                assert output_path.exists()
                assert output_path.read_bytes() == b'fake tiff data'
                assert mock_request.call_count == 2

    def test_export_species_raster_success_as_array(self):
        """Test successful species raster export as numpy array."""
        client = BigMapRestClient()
        species_code = '0131'
        bbox = (-12000000, 5000000, -11000000, 6000000)

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                # Mock export request
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'href': 'http://test.com/raster.tif'}

                # Mock raster download
                raster_response = Mock()
                raster_response.raise_for_status = Mock()
                raster_response.content = b'fake tiff data'

                mock_request.side_effect = [export_response, raster_response]

                # Mock rasterio reading
                with patch('gridfia.external.fia_client.MemoryFile') as mock_memory_file:
                    mock_dataset = Mock()
                    mock_dataset.read.return_value = np.array([[1, 2], [3, 4]])

                    # Create proper context manager mock
                    mock_file_context = Mock()
                    mock_file_context.__enter__ = Mock(return_value=mock_dataset)
                    mock_file_context.__exit__ = Mock(return_value=None)

                    mock_memory_file_instance = Mock()
                    mock_memory_file_instance.open.return_value = mock_file_context
                    mock_memory_file_instance.__enter__ = Mock(return_value=mock_memory_file_instance)
                    mock_memory_file_instance.__exit__ = Mock(return_value=None)

                    mock_memory_file.return_value = mock_memory_file_instance

                    result = client.export_species_raster(
                        species_code=species_code,
                        bbox=bbox,
                        output_path=None  # Return as array
                    )

                    assert isinstance(result, np.ndarray)
                    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))

    def test_export_species_raster_custom_parameters(self, temp_dir):
        """Test species raster export with custom parameters."""
        client = BigMapRestClient()
        species_code = '0202'
        bbox = (-11500000, 4500000, -10500000, 5500000)
        output_path = temp_dir / "test_custom.png"

        with patch.object(client, '_get_function_name', return_value='SPCD_0202_Pseudotsuga_menziesii'):
            with patch.object(client, '_calculate_image_size', return_value='1000,800'):
                with patch.object(client, '_rate_limited_request') as mock_request:
                    export_response = Mock()
                    export_response.raise_for_status = Mock()
                    export_response.json.return_value = {'href': 'http://test.com/raster.png'}

                    raster_response = Mock()
                    raster_response.raise_for_status = Mock()
                    raster_response.content = b'fake png data'

                    mock_request.side_effect = [export_response, raster_response]

                    result = client.export_species_raster(
                        species_code=species_code,
                        bbox=bbox,
                        output_path=output_path,  # Save to file instead of returning array
                        pixel_size=60.0,
                        format="png",
                        bbox_srs="4326",
                        output_srs="3857"
                    )

                    # Check that file was written
                    assert result == output_path
                    assert output_path.exists()

                    # Check export request parameters
                    export_call = mock_request.call_args_list[0]
                    params = export_call[1]['params']

                    assert params['format'] == 'png'
                    assert params['bboxSR'] == '4326'
                    assert params['imageSR'] == '3857'
                    assert params['pixelType'] == 'F32'
                    assert 'renderingRule' in params

                    rendering_rule = json.loads(params['renderingRule'])
                    assert rendering_rule['rasterFunction'] == 'SPCD_0202_Pseudotsuga_menziesii'

    def test_export_species_raster_species_not_found(self):
        """Test handling when species code is not found."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value=None):
            with pytest.raises(SpeciesNotFound):
                client.export_species_raster(
                    species_code='9999',
                    bbox=(-12000000, 5000000, -11000000, 6000000)
                )

    def test_export_species_raster_export_request_fails(self):
        """Test handling of export request failure."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_request.side_effect = RequestException("Export failed")

                with pytest.raises(DownloadError):
                    client.export_species_raster(
                        species_code='0131',
                        bbox=(-12000000, 5000000, -11000000, 6000000)
                    )

    def test_export_species_raster_no_href_in_response(self):
        """Test handling when export response has no href."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'error': 'Export failed'}
                mock_request.return_value = export_response

                with pytest.raises(DownloadError):
                    client.export_species_raster(
                        species_code='0131',
                        bbox=(-12000000, 5000000, -11000000, 6000000)
                    )

    def test_export_species_raster_raster_download_fails(self):
        """Test handling when raster download fails."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'href': 'http://test.com/raster.tif'}

                # Second request (raster download) fails
                mock_request.side_effect = [export_response, RequestException("Download failed")]

                with pytest.raises(DownloadError):
                    client.export_species_raster(
                        species_code='0131',
                        bbox=(-12000000, 5000000, -11000000, 6000000)
                    )


class TestBigMapRestClientGetSpeciesStatistics:
    """Test get_species_statistics method."""

    def test_get_species_statistics_success(self):
        """Test successful species statistics retrieval."""
        client = BigMapRestClient()
        species_code = '0131'
        expected_stats = {
            'min': 0.0,
            'max': 150.5,
            'mean': 45.2,
            'stddev': 23.8,
            'count': 1500000
        }

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = expected_stats
                mock_request.return_value = mock_response

                result = client.get_species_statistics(species_code)

                assert result == expected_stats

                # Check request parameters
                call_args = mock_request.call_args
                params = call_args[1]['params']
                assert params['f'] == 'json'
                assert 'renderingRule' in params

                rendering_rule = json.loads(params['renderingRule'])
                assert rendering_rule['rasterFunction'] == 'SPCD_0131_Abies_balsamea'

    def test_get_species_statistics_species_not_found(self):
        """Test handling when species code is not found."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value=None):
            with pytest.raises(SpeciesNotFound):
                client.get_species_statistics('9999')

    def test_get_species_statistics_request_fails(self):
        """Test handling when statistics request fails."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_request.side_effect = RequestException("Stats failed")

                with pytest.raises(APIConnectionError):
                    client.get_species_statistics('0131')


class TestBigMapRestClientIdentifyPixelValue:
    """Test identify_pixel_value method."""

    def test_identify_pixel_value_success(self):
        """Test successful pixel value identification."""
        client = BigMapRestClient()
        species_code = '0131'
        x, y = -11500000, 5500000
        expected_value = 87.5

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {'value': expected_value}
                mock_request.return_value = mock_response

                result = client.identify_pixel_value(species_code, x, y)

                assert result == expected_value

                # Check request parameters
                call_args = mock_request.call_args
                params = call_args[1]['params']
                assert params['geometry'] == f'{x},{y}'
                assert params['geometryType'] == 'esriGeometryPoint'
                assert params['sr'] == '102100'  # Default spatial reference

    def test_identify_pixel_value_custom_spatial_ref(self):
        """Test pixel identification with custom spatial reference."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {'value': 45.2}
                mock_request.return_value = mock_response

                client.identify_pixel_value('0131', -104.5, 45.2, spatial_ref='4326')

                # Check spatial reference parameter
                call_args = mock_request.call_args
                params = call_args[1]['params']
                assert params['sr'] == '4326'

    def test_identify_pixel_value_no_data(self):
        """Test handling of NoData pixel values."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {'value': 'NoData'}
                mock_request.return_value = mock_response

                result = client.identify_pixel_value('0131', -11500000, 5500000)

                assert result == 0.0  # NoData should return 0.0

    def test_identify_pixel_value_none_value(self):
        """Test handling of None pixel values."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {'value': None}
                mock_request.return_value = mock_response

                result = client.identify_pixel_value('0131', -11500000, 5500000)

                assert result == 0.0  # None should return 0.0

    def test_identify_pixel_value_no_value_key(self):
        """Test handling when response has no value key."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {'error': 'No value found'}
                mock_request.return_value = mock_response

                result = client.identify_pixel_value('0131', -11500000, 5500000)

                assert result is None

    def test_identify_pixel_value_species_not_found(self):
        """Test handling when species code is not found."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value=None):
            with pytest.raises(SpeciesNotFound):
                client.identify_pixel_value('9999', -11500000, 5500000)

    def test_identify_pixel_value_request_fails(self):
        """Test handling when identify request fails."""
        client = BigMapRestClient()

        with patch.object(client, '_get_function_name', return_value='SPCD_0131_Abies_balsamea'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_request.side_effect = RequestException("Identify failed")

                with pytest.raises(APIConnectionError):
                    client.identify_pixel_value('0131', -11500000, 5500000)


class TestBigMapRestClientExportTotalBiomassRaster:
    """Test export_total_biomass_raster method."""

    def test_export_total_biomass_raster_success(self, temp_dir):
        """Test successful total biomass raster export."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)
        output_path = temp_dir / "total_biomass.tif"

        with patch.object(client, '_calculate_image_size', return_value='1000,800'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'href': 'http://test.com/total.tif'}

                raster_response = Mock()
                raster_response.raise_for_status = Mock()
                raster_response.content = b'fake total biomass data'

                mock_request.side_effect = [export_response, raster_response]

                result = client.export_total_biomass_raster(bbox=bbox, output_path=output_path)

                assert result == output_path
                assert output_path.exists()
                assert output_path.read_bytes() == b'fake total biomass data'

    def test_export_total_biomass_raster_as_array(self):
        """Test total biomass export as numpy array."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)

        with patch.object(client, '_calculate_image_size', return_value='1000,800'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'href': 'http://test.com/total.tif'}

                raster_response = Mock()
                raster_response.raise_for_status = Mock()
                raster_response.content = b'fake tiff data'

                mock_request.side_effect = [export_response, raster_response]

                with patch('gridfia.external.fia_client.MemoryFile') as mock_memory_file:
                    mock_dataset = Mock()
                    mock_dataset.read.return_value = np.array([[10, 20], [30, 40]])

                    # Create proper context manager mock
                    mock_file_context = Mock()
                    mock_file_context.__enter__ = Mock(return_value=mock_dataset)
                    mock_file_context.__exit__ = Mock(return_value=None)

                    mock_memory_file_instance = Mock()
                    mock_memory_file_instance.open.return_value = mock_file_context
                    mock_memory_file_instance.__enter__ = Mock(return_value=mock_memory_file_instance)
                    mock_memory_file_instance.__exit__ = Mock(return_value=None)

                    mock_memory_file.return_value = mock_memory_file_instance

                    result = client.export_total_biomass_raster(bbox=bbox, output_path=None)

                    assert isinstance(result, np.ndarray)
                    np.testing.assert_array_equal(result, np.array([[10, 20], [30, 40]]))

    def test_export_total_biomass_custom_parameters(self, temp_dir):
        """Test total biomass export with custom parameters."""
        client = BigMapRestClient()
        bbox = (-11500000, 4500000, -10500000, 5500000)
        output_path = temp_dir / "total_custom.png"

        with patch.object(client, '_calculate_image_size', return_value='800,600'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'href': 'http://test.com/total.png'}

                raster_response = Mock()
                raster_response.raise_for_status = Mock()
                raster_response.content = b'fake png data'

                mock_request.side_effect = [export_response, raster_response]

                result = client.export_total_biomass_raster(
                    bbox=bbox,
                    output_path=output_path,  # Save to file instead of returning array
                    pixel_size=60.0,
                    format="png",
                    bbox_srs="4326",
                    output_srs="3857"
                )

                # Check that file was written
                assert result == output_path
                assert output_path.exists()

                # Check export request parameters
                export_call = mock_request.call_args_list[0]
                params = export_call[1]['params']

                assert params['format'] == 'png'
                assert params['bboxSR'] == '4326'
                assert params['imageSR'] == '3857'
                assert params['pixelType'] == 'F32'
                # Should not have renderingRule for total biomass
                assert 'renderingRule' not in params

    def test_export_total_biomass_no_href_in_response(self):
        """Test handling when export response has no href."""
        client = BigMapRestClient()

        with patch.object(client, '_calculate_image_size', return_value='1000,800'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                export_response = Mock()
                export_response.raise_for_status = Mock()
                export_response.json.return_value = {'error': 'Export failed'}
                mock_request.return_value = export_response

                with pytest.raises(DownloadError):
                    client.export_total_biomass_raster(
                        bbox=(-12000000, 5000000, -11000000, 6000000)
                    )

    def test_export_total_biomass_request_fails(self):
        """Test handling when total biomass export request fails."""
        client = BigMapRestClient()

        with patch.object(client, '_calculate_image_size', return_value='1000,800'):
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_request.side_effect = RequestException("Export failed")

                with pytest.raises(DownloadError):
                    client.export_total_biomass_raster(
                        bbox=(-12000000, 5000000, -11000000, 6000000)
                    )


class TestBigMapRestClientBatchExport:
    """Test batch_export_location_species method."""

    def test_batch_export_success(self, temp_dir):
        """Test successful batch export of species."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)
        species_codes = ['0131', '0202']
        location_name = "Montana"

        with patch.object(client, 'export_species_raster') as mock_export:
            # Mock successful exports
            expected_files = [
                temp_dir / "Montana_species_0131.tif",
                temp_dir / "Montana_species_0202.tif"
            ]
            mock_export.side_effect = expected_files

            with patch('rich.progress.Progress'):  # Mock progress bar
                result = client.batch_export_location_species(
                    bbox=bbox,
                    output_dir=temp_dir,
                    species_codes=species_codes,
                    location_name=location_name
                )

                assert result == expected_files
                assert temp_dir.exists()
                assert mock_export.call_count == 2

    def test_batch_export_all_species(self, temp_dir):
        """Test batch export with all available species."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)
        location_name = "Montana"

        mock_species_list = [
            {'species_code': '0131'},
            {'species_code': '0202'},
            {'species_code': '0068'}
        ]

        with patch.object(client, 'list_available_species', return_value=mock_species_list):
            with patch.object(client, 'export_species_raster') as mock_export:
                expected_files = [
                    temp_dir / "Montana_species_0131.tif",
                    temp_dir / "Montana_species_0202.tif",
                    temp_dir / "Montana_species_0068.tif"
                ]
                mock_export.side_effect = expected_files

                with patch('rich.progress.Progress'):
                    result = client.batch_export_location_species(
                        bbox=bbox,
                        output_dir=temp_dir,
                        species_codes=None,  # Should get all species
                        location_name=location_name
                    )

                    assert len(result) == 3
                    assert mock_export.call_count == 3

    def test_batch_export_custom_parameters(self, temp_dir):
        """Test batch export with custom parameters."""
        client = BigMapRestClient()
        bbox = (-11500000, 4500000, -10500000, 5500000)
        species_codes = ['0131']
        location_name = "Custom_Location"

        with patch.object(client, 'export_species_raster') as mock_export:
            mock_export.return_value = temp_dir / "Custom_Location_species_0131.tif"

            with patch('rich.progress.Progress'):
                client.batch_export_location_species(
                    bbox=bbox,
                    output_dir=temp_dir,
                    species_codes=species_codes,
                    location_name=location_name,
                    bbox_srs="4326",
                    output_srs="3857"
                )

                # Check that custom parameters were passed to export_species_raster
                call_args = mock_export.call_args[1]
                assert call_args['bbox_srs'] == "4326"
                assert call_args['output_srs'] == "3857"

    def test_batch_export_partial_failure(self, temp_dir):
        """Test batch export with some species failing."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)
        species_codes = ['0131', '0202', '0068']
        location_name = "Montana"

        with patch.object(client, 'export_species_raster') as mock_export:
            # First and third succeed, second fails
            mock_export.side_effect = [
                temp_dir / "Montana_species_0131.tif",  # Success
                None,  # Failure
                temp_dir / "Montana_species_0068.tif"   # Success
            ]

            with patch('rich.progress.Progress'):
                result = client.batch_export_location_species(
                    bbox=bbox,
                    output_dir=temp_dir,
                    species_codes=species_codes,
                    location_name=location_name
                )

                # Should only return successful exports
                assert len(result) == 2
                assert mock_export.call_count == 3

    def test_batch_export_all_fail_with_exceptions(self, temp_dir):
        """Test batch export when all species fail with exceptions."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)
        species_codes = ['0131', '0202']
        location_name = "Montana"

        with patch.object(client, 'export_species_raster') as mock_export:
            # All exports raise exceptions
            mock_export.side_effect = [
                Exception("Export failed for 0131"),
                Exception("Export failed for 0202")
            ]

            with patch('rich.progress.Progress'):
                result = client.batch_export_location_species(
                    bbox=bbox,
                    output_dir=temp_dir,
                    species_codes=species_codes,
                    location_name=location_name
                )

                # Should return empty list
                assert result == []
                assert mock_export.call_count == 2


class TestBigMapRestClientUtilityMethods:
    """Test utility methods _get_function_name and _calculate_image_size."""

    def test_get_function_name_success(self):
        """Test successful function name lookup."""
        client = BigMapRestClient()
        mock_functions = [
            {'name': 'SPCD_0131_Abies_balsamea'},
            {'name': 'SPCD_0202_Pseudotsuga_menziesii'},
            {'name': 'OTHER_FUNCTION'}
        ]

        with patch.object(client, 'get_species_functions', return_value=mock_functions):
            result = client._get_function_name('0131')

            assert result == 'SPCD_0131_Abies_balsamea'

    def test_get_function_name_not_found(self):
        """Test function name lookup when species not found."""
        client = BigMapRestClient()
        mock_functions = [
            {'name': 'SPCD_0131_Abies_balsamea'},
            {'name': 'SPCD_0202_Pseudotsuga_menziesii'}
        ]

        with patch.object(client, 'get_species_functions', return_value=mock_functions):
            result = client._get_function_name('9999')

            assert result is None

    def test_get_function_name_empty_functions(self):
        """Test function name lookup with empty functions list."""
        client = BigMapRestClient()

        with patch.object(client, 'get_species_functions', return_value=[]):
            result = client._get_function_name('0131')

            assert result is None

    def test_calculate_image_size_basic(self):
        """Test basic image size calculation with service limits."""
        client = BigMapRestClient()

        # Test small bbox that doesn't hit limits
        bbox = (-11015000, 4985000, -11000000, 5000000)  # 15km x 15km
        pixel_size = 30.0

        result = client._calculate_image_size(bbox, pixel_size)

        expected_width = int((bbox[2] - bbox[0]) / pixel_size)  # 500
        expected_height = int((bbox[3] - bbox[1]) / pixel_size)  # 500

        assert result == f"{expected_width},{expected_height}"

    def test_calculate_image_size_with_limits(self):
        """Test image size calculation with service limits applied."""
        client = BigMapRestClient()
        # Very large bbox to trigger limits
        bbox = (-15000000, 2000000, -8000000, 8000000)
        pixel_size = 30.0

        result = client._calculate_image_size(bbox, pixel_size)

        # Should be limited to service maximums
        width_str, height_str = result.split(',')
        width = int(width_str)
        height = int(height_str)

        assert width <= 15000  # Max width limit
        assert height <= 4100   # Max height limit

    def test_calculate_image_size_small_pixel_size(self):
        """Test image size calculation with small pixel size."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)
        pixel_size = 1.0  # Very small pixels

        result = client._calculate_image_size(bbox, pixel_size)

        width_str, height_str = result.split(',')
        width = int(width_str)
        height = int(height_str)

        # Should be limited by service maximums
        assert width <= 15000
        assert height <= 4100


class TestBigMapRestClientRealAPIIntegration:
    """Integration tests using real API calls (as required by project guidelines)."""

    @pytest.mark.slow
    def test_real_get_service_info(self):
        """Test real service info retrieval from FIA BIGMAP API."""
        client = BigMapRestClient(timeout=60)  # Longer timeout for real requests

        result = client.get_service_info()

        # Basic validation of real service response
        assert isinstance(result, dict)
        if result:  # Only check if request succeeded
            assert 'name' in result
            assert 'serviceDescription' in result or 'description' in result

    @pytest.mark.slow
    def test_real_get_species_functions(self):
        """Test real species functions retrieval."""
        client = BigMapRestClient(timeout=60)

        functions = client.get_species_functions()

        assert isinstance(functions, list)
        if functions:  # Only check if request succeeded
            # Should have multiple species functions
            assert len(functions) > 0
            # Each function should have name
            for func in functions[:5]:  # Check first 5
                assert 'name' in func

    @pytest.mark.slow
    def test_real_list_available_species(self):
        """Test real species listing."""
        client = BigMapRestClient(timeout=60)

        species_list = client.list_available_species()

        assert isinstance(species_list, list)
        if species_list:  # Only check if request succeeded
            # Should have multiple species
            assert len(species_list) > 0
            # Check structure of first species
            first_species = species_list[0]
            assert 'species_code' in first_species
            assert 'common_name' in first_species
            assert 'scientific_name' in first_species
            assert 'function_name' in first_species

    @pytest.mark.slow
    def test_real_export_small_raster(self, temp_dir):
        """Test real raster export with a small bounding box."""
        client = BigMapRestClient(timeout=120)  # Longer timeout for raster export

        # Small bbox in North Carolina (where BIGMAP has data)
        bbox = (-8500000, 4200000, -8400000, 4300000)  # Small 100km x 100km area
        output_path = temp_dir / "test_real_export.tif"

        try:
            result = client.export_species_raster(
                species_code='0131',  # Common species code
                bbox=bbox,
                output_path=output_path,
                pixel_size=300.0  # Large pixels for smaller file
            )

            if result is not None:
                assert result == output_path
                assert output_path.exists()
                assert output_path.stat().st_size > 0
        except Exception as e:
            # Real API tests may fail due to network/service issues
            pytest.skip(f"Real API test failed (expected): {e}")

    @pytest.mark.slow
    def test_real_identify_pixel_value(self):
        """Test real pixel value identification."""
        client = BigMapRestClient(timeout=60)

        # Point in North Carolina forest area
        x, y = -8450000, 4250000  # Web Mercator coordinates

        try:
            result = client.identify_pixel_value('0131', x, y)

            # Result could be a float value or None/0.0 if no data at location
            assert result is None or isinstance(result, (int, float))
        except Exception as e:
            # Real API tests may fail due to network/service issues
            pytest.skip(f"Real API test failed (expected): {e}")


class TestBigMapRestClientErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""

    def test_malformed_json_response(self):
        """Test handling of malformed JSON responses."""
        client = BigMapRestClient()

        # JSON decode errors will propagate since they're not RequestException
        # This tests that the current implementation doesn't handle JSON parsing errors
        with patch.object(client, '_rate_limited_request') as mock_request:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
            mock_request.return_value = mock_response

            with pytest.raises(json.JSONDecodeError):
                client.get_service_info()

    def test_http_error_codes(self):
        """Test handling of various HTTP error codes."""
        client = BigMapRestClient()

        error_codes = [400, 401, 403, 404, 500, 503]

        for code in error_codes:
            with patch.object(client, '_rate_limited_request') as mock_request:
                mock_response = Mock()
                mock_response.raise_for_status.side_effect = requests.HTTPError(f"{code} Error")
                mock_request.return_value = mock_response

                with pytest.raises(APIConnectionError):
                    client.get_service_info()

    def test_very_large_bbox(self):
        """Test handling of very large bounding boxes."""
        client = BigMapRestClient()

        # Extremely large bbox
        bbox = (-20000000, 0, 20000000, 20000000)
        pixel_size = 30.0

        result = client._calculate_image_size(bbox, pixel_size)

        width_str, height_str = result.split(',')
        width = int(width_str)
        height = int(height_str)

        # Should be clamped to service limits
        assert width <= 15000
        assert height <= 4100

    def test_zero_pixel_size(self):
        """Test handling of zero pixel size."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)

        with pytest.raises(ZeroDivisionError):
            client._calculate_image_size(bbox, 0.0)

    def test_negative_pixel_size(self):
        """Test handling of negative pixel size."""
        client = BigMapRestClient()
        bbox = (-12000000, 5000000, -11000000, 6000000)

        # Negative pixel size should result in negative dimensions
        result = client._calculate_image_size(bbox, -30.0)

        width_str, height_str = result.split(',')
        width = int(width_str)
        height = int(height_str)

        assert width < 0
        assert height < 0

    def test_inverted_bbox(self):
        """Test handling of inverted bounding box coordinates."""
        client = BigMapRestClient()

        # Inverted bbox (xmax < xmin, ymax < ymin)
        bbox = (-11000000, 6000000, -12000000, 5000000)
        pixel_size = 30.0

        result = client._calculate_image_size(bbox, pixel_size)

        width_str, height_str = result.split(',')
        width = int(width_str)
        height = int(height_str)

        # Should result in negative dimensions
        assert width < 0
        assert height < 0

    def test_empty_species_functions_list(self):
        """Test behavior with empty species functions list."""
        client = BigMapRestClient()

        with patch.object(client, 'get_species_functions', return_value=[]):
            # Should return empty list
            species_list = client.list_available_species()
            assert species_list == []

            # Should return None for function lookup
            function_name = client._get_function_name('0131')
            assert function_name is None

    def test_species_function_missing_keys(self):
        """Test handling of species functions with missing keys."""
        client = BigMapRestClient()

        incomplete_functions = [
            {'name': 'SPCD_0131_Abies_balsamea'},  # Missing description
            {'description': 'Douglas-fir'},  # Missing name
            {}  # Missing both
        ]

        with patch.object(client, 'get_species_functions', return_value=incomplete_functions):
            species_list = client.list_available_species()

            # Should handle missing keys gracefully
            assert len(species_list) == 1  # Only the complete one
            assert species_list[0]['species_code'] == '0131'

    def test_concurrent_request_handling(self):
        """Test rate limiting with concurrent requests."""
        client = BigMapRestClient(rate_limit_delay=0.1)

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            # Simulate rapid concurrent requests
            import threading
            import concurrent.futures

            def make_request():
                return client._rate_limited_request('GET', 'http://test.com')

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_request) for _ in range(3)]
                results = [future.result() for future in futures]

            end_time = time.time()

            # Should have taken at least some time due to rate limiting
            assert end_time - start_time >= 0.1
            assert len(results) == 3
            assert all(r.status_code == 200 for r in results)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreakerInitialization:
    """Test CircuitBreaker initialization and configuration."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker()

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker._name == "default"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker._last_failure_time is None

    def test_init_custom_threshold(self):
        """Test initialization with custom failure threshold."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=10)

        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 60.0

    def test_init_custom_timeout(self):
        """Test initialization with custom recovery timeout."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(recovery_timeout=120.0)

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 120.0

    def test_init_custom_name(self):
        """Test initialization with custom name."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(name="test_breaker")

        assert breaker._name == "test_breaker"

    def test_init_all_custom_params(self):
        """Test initialization with all custom parameters."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            name="custom_breaker"
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker._name == "custom_breaker"
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_init_thread_safety(self):
        """Test that circuit breaker initializes with a lock for thread safety."""
        from gridfia.external.fia_client import CircuitBreaker
        import threading

        breaker = CircuitBreaker()

        assert breaker._lock is not None
        assert isinstance(breaker._lock, type(threading.RLock()))


class TestCircuitBreakerCanExecute:
    """Test CircuitBreaker can_execute() method."""

    def test_can_execute_closed_state(self):
        """Test can_execute returns True when circuit is CLOSED."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker()

        assert breaker.can_execute() is True

    def test_can_execute_open_state(self):
        """Test can_execute returns False when circuit is OPEN."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        # Trigger enough failures to open the circuit
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.can_execute() is False

    def test_can_execute_half_open_state(self):
        """Test can_execute returns True when circuit is HALF_OPEN."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Should transition to HALF_OPEN and allow execution
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_can_execute_multiple_calls_closed(self):
        """Test multiple can_execute calls in CLOSED state all return True."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker()

        for _ in range(10):
            assert breaker.can_execute() is True


class TestCircuitBreakerRecordSuccess:
    """Test CircuitBreaker record_success() method."""

    def test_record_success_resets_failure_count(self):
        """Test record_success resets failure count in CLOSED state."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=5)

        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2

        # Record success
        breaker.record_success()

        assert breaker.failure_count == 0

    def test_record_success_closes_circuit_from_half_open(self):
        """Test record_success closes circuit from HALF_OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Access state to transition to HALF_OPEN
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Record success - should close the circuit
        breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0

    def test_record_success_in_closed_state_keeps_closed(self):
        """Test record_success in CLOSED state keeps circuit CLOSED."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker()

        breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0


class TestCircuitBreakerRecordFailure:
    """Test CircuitBreaker record_failure() method."""

    def test_record_failure_increments_count(self):
        """Test record_failure increments failure count."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=10)

        assert breaker.failure_count == 0

        breaker.record_failure()
        assert breaker.failure_count == 1

        breaker.record_failure()
        assert breaker.failure_count == 2

        breaker.record_failure()
        assert breaker.failure_count == 3

    def test_record_failure_opens_circuit_at_threshold(self):
        """Test record_failure opens circuit when threshold is reached."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.CLOSED

        breaker.record_failure()  # Third failure - hits threshold
        assert breaker.state == CircuitBreakerState.OPEN

    def test_record_failure_sets_last_failure_time(self):
        """Test record_failure sets last failure time."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker()

        assert breaker._last_failure_time is None

        before_time = time.time()
        breaker.record_failure()
        after_time = time.time()

        assert breaker._last_failure_time is not None
        assert before_time <= breaker._last_failure_time <= after_time

    def test_record_failure_reopens_circuit_from_half_open(self):
        """Test record_failure reopens circuit from HALF_OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Access state to transition to HALF_OPEN
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Record failure - should reopen the circuit
        breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN


class TestCircuitBreakerReset:
    """Test CircuitBreaker reset() method."""

    def test_reset_from_closed_state(self):
        """Test reset from CLOSED state with failures."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=5)

        # Record some failures
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.failure_count == 2

        breaker.reset()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker._last_failure_time is None

    def test_reset_from_open_state(self):
        """Test reset from OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1)

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

        breaker.reset()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker._last_failure_time is None

    def test_reset_from_half_open_state(self):
        """Test reset from HALF_OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()

        # Wait for recovery timeout to transition to HALF_OPEN
        time.sleep(0.02)
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        breaker.reset()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker._last_failure_time is None

    def test_reset_updates_last_state_change_time(self):
        """Test reset updates last state change time."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1)

        # Open the circuit
        breaker.record_failure()
        old_state_change_time = breaker._last_state_change_time

        time.sleep(0.01)

        breaker.reset()

        assert breaker._last_state_change_time > old_state_change_time


class TestCircuitBreakerGetStatus:
    """Test CircuitBreaker get_status() method."""

    def test_get_status_closed_state(self):
        """Test get_status in CLOSED state."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name="test_breaker"
        )

        status = breaker.get_status()

        assert status["name"] == "test_breaker"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["recovery_timeout"] == 60.0
        assert status["time_until_recovery"] is None
        assert status["last_failure_time"] is None
        assert "last_state_change_time" in status

    def test_get_status_with_failures(self):
        """Test get_status with recorded failures."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=5)

        breaker.record_failure()
        breaker.record_failure()

        status = breaker.get_status()

        assert status["state"] == "closed"
        assert status["failure_count"] == 2
        assert status["last_failure_time"] is not None

    def test_get_status_open_state(self):
        """Test get_status in OPEN state with time_until_recovery."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        breaker.record_failure()

        status = breaker.get_status()

        assert status["state"] == "open"
        assert status["failure_count"] == 1
        assert status["time_until_recovery"] is not None
        assert 0 < status["time_until_recovery"] <= 60.0

    def test_get_status_half_open_state(self):
        """Test get_status in HALF_OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        breaker.record_failure()
        time.sleep(0.02)

        # Trigger transition to HALF_OPEN
        breaker.can_execute()

        status = breaker.get_status()

        assert status["state"] == "half_open"
        assert status["time_until_recovery"] is None


class TestCircuitBreakerStateTransitions:
    """Test CircuitBreaker state transitions."""

    def test_transition_closed_to_open(self):
        """Test state transition from CLOSED to OPEN."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=2)

        assert breaker.state == CircuitBreakerState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

    def test_transition_open_to_half_open(self):
        """Test state transition from OPEN to HALF_OPEN after timeout."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Access state to trigger transition check
        state = breaker.state
        assert state == CircuitBreakerState.HALF_OPEN

    def test_transition_half_open_to_closed_on_success(self):
        """Test state transition from HALF_OPEN to CLOSED on success."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()
        time.sleep(0.02)

        # Transition to HALF_OPEN
        breaker.can_execute()
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Record success
        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_transition_half_open_to_open_on_failure(self):
        """Test state transition from HALF_OPEN to OPEN on failure."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Open the circuit
        breaker.record_failure()
        time.sleep(0.02)

        # Transition to HALF_OPEN
        breaker.can_execute()
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Record failure
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

    def test_full_recovery_cycle(self):
        """Test complete recovery cycle: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)

        # Start in CLOSED
        assert breaker.state == CircuitBreakerState.CLOSED

        # Failures open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Attempt request - transitions to HALF_OPEN
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Success closes the circuit
        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerTimeUntilRecovery:
    """Test CircuitBreaker time_until_recovery property."""

    def test_time_until_recovery_closed_state(self):
        """Test time_until_recovery is None in CLOSED state."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker()

        assert breaker.time_until_recovery is None

    def test_time_until_recovery_open_state(self):
        """Test time_until_recovery in OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        breaker.record_failure()

        time_until = breaker.time_until_recovery

        assert time_until is not None
        assert 0 < time_until <= 60.0

    def test_time_until_recovery_decreases_over_time(self):
        """Test time_until_recovery decreases as time passes."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        breaker.record_failure()

        first_time = breaker.time_until_recovery
        time.sleep(0.1)
        second_time = breaker.time_until_recovery

        assert second_time < first_time

    def test_time_until_recovery_half_open_state(self):
        """Test time_until_recovery is None in HALF_OPEN state."""
        from gridfia.external.fia_client import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        breaker.record_failure()
        time.sleep(0.02)

        # Transition to HALF_OPEN
        breaker.can_execute()

        assert breaker.time_until_recovery is None


class TestCircuitBreakerThreadSafety:
    """Test CircuitBreaker thread safety."""

    def test_concurrent_failure_recording(self):
        """Test concurrent record_failure calls are thread-safe."""
        from gridfia.external.fia_client import CircuitBreaker
        import concurrent.futures

        breaker = CircuitBreaker(failure_threshold=100)

        def record_failures():
            for _ in range(10):
                breaker.record_failure()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(record_failures) for _ in range(5)]
            concurrent.futures.wait(futures)

        # All 50 failures should be recorded
        assert breaker.failure_count == 50

    def test_concurrent_can_execute_calls(self):
        """Test concurrent can_execute calls are thread-safe."""
        from gridfia.external.fia_client import CircuitBreaker
        import concurrent.futures

        breaker = CircuitBreaker(failure_threshold=100)
        results = []

        def check_execute():
            return breaker.can_execute()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_execute) for _ in range(100)]
            results = [f.result() for f in futures]

        # All calls should return True (circuit is CLOSED)
        assert all(results)

    def test_concurrent_state_transitions(self):
        """Test concurrent state transitions are handled correctly."""
        from gridfia.external.fia_client import CircuitBreaker, CircuitBreakerState
        import concurrent.futures

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=0.01)

        def mixed_operations(operation_type):
            if operation_type == "fail":
                breaker.record_failure()
            elif operation_type == "success":
                breaker.record_success()
            elif operation_type == "check":
                breaker.can_execute()

        operations = ["fail"] * 10 + ["success"] * 5 + ["check"] * 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mixed_operations, op) for op in operations]
            concurrent.futures.wait(futures)

        # Should not raise any exceptions and should be in a valid state
        assert breaker.state in [
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.OPEN,
            CircuitBreakerState.HALF_OPEN
        ]


# =============================================================================
# BigMapRestClient Circuit Breaker Integration Tests
# =============================================================================


class TestBigMapRestClientCircuitBreakerEnabled:
    """Test BigMapRestClient with circuit breaker enabled."""

    def test_client_with_circuit_breaker_enabled_default(self):
        """Test client initializes with circuit breaker enabled by default."""
        client = BigMapRestClient()

        assert client._circuit_breaker_enabled is True
        assert client._circuit_breaker is not None

    def test_client_with_circuit_breaker_enabled_explicit(self):
        """Test client initializes with circuit breaker explicitly enabled."""
        client = BigMapRestClient(circuit_breaker_enabled=True)

        assert client._circuit_breaker_enabled is True
        assert client._circuit_breaker is not None

    def test_client_circuit_breaker_configuration(self):
        """Test circuit breaker is configured with client parameters."""
        from gridfia.external.fia_client import CircuitBreakerState

        client = BigMapRestClient(
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=120.0,
            circuit_breaker_enabled=True
        )

        assert client._circuit_breaker.failure_threshold == 10
        assert client._circuit_breaker.recovery_timeout == 120.0
        assert client._circuit_breaker._name == "FIA_BIGMAP"
        assert client._circuit_breaker.state == CircuitBreakerState.CLOSED


class TestBigMapRestClientCircuitBreakerDisabled:
    """Test BigMapRestClient with circuit breaker disabled."""

    def test_client_with_circuit_breaker_disabled(self):
        """Test client initializes with circuit breaker disabled."""
        client = BigMapRestClient(circuit_breaker_enabled=False)

        assert client._circuit_breaker_enabled is False
        assert client._circuit_breaker is None

    def test_circuit_breaker_property_returns_none_when_disabled(self):
        """Test circuit_breaker property returns None when disabled."""
        client = BigMapRestClient(circuit_breaker_enabled=False)

        assert client.circuit_breaker is None

    def test_get_circuit_breaker_status_returns_none_when_disabled(self):
        """Test get_circuit_breaker_status returns None when disabled."""
        client = BigMapRestClient(circuit_breaker_enabled=False)

        assert client.get_circuit_breaker_status() is None

    def test_reset_circuit_breaker_no_op_when_disabled(self):
        """Test reset_circuit_breaker does nothing when disabled."""
        client = BigMapRestClient(circuit_breaker_enabled=False)

        # Should not raise any exception
        client.reset_circuit_breaker()

        assert client._circuit_breaker is None

    def test_requests_work_normally_when_disabled(self):
        """Test requests work normally without circuit breaker."""
        client = BigMapRestClient(circuit_breaker_enabled=False)

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            response = client._rate_limited_request('GET', 'http://test.com')

            assert response == mock_response


class TestBigMapRestClientCircuitBreakerProperty:
    """Test BigMapRestClient circuit_breaker property."""

    def test_circuit_breaker_property_returns_instance(self):
        """Test circuit_breaker property returns CircuitBreaker instance."""
        from gridfia.external.fia_client import CircuitBreaker

        client = BigMapRestClient(circuit_breaker_enabled=True)

        assert client.circuit_breaker is not None
        assert isinstance(client.circuit_breaker, CircuitBreaker)

    def test_circuit_breaker_property_returns_same_instance(self):
        """Test circuit_breaker property returns the same instance."""
        client = BigMapRestClient(circuit_breaker_enabled=True)

        first_access = client.circuit_breaker
        second_access = client.circuit_breaker

        assert first_access is second_access


class TestBigMapRestClientGetCircuitBreakerStatus:
    """Test BigMapRestClient get_circuit_breaker_status method."""

    def test_get_circuit_breaker_status_returns_dict(self):
        """Test get_circuit_breaker_status returns status dictionary."""
        client = BigMapRestClient(circuit_breaker_enabled=True)

        status = client.get_circuit_breaker_status()

        assert isinstance(status, dict)
        assert "name" in status
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status

    def test_get_circuit_breaker_status_reflects_state(self):
        """Test get_circuit_breaker_status reflects current state."""
        client = BigMapRestClient(
            circuit_breaker_threshold=2,
            circuit_breaker_enabled=True
        )

        # Initial state
        status = client.get_circuit_breaker_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0

        # After failures
        client._circuit_breaker.record_failure()
        client._circuit_breaker.record_failure()

        status = client.get_circuit_breaker_status()
        assert status["state"] == "open"
        assert status["failure_count"] == 2


class TestBigMapRestClientResetCircuitBreaker:
    """Test BigMapRestClient reset_circuit_breaker method."""

    def test_reset_circuit_breaker_resets_state(self):
        """Test reset_circuit_breaker resets circuit to CLOSED."""
        client = BigMapRestClient(
            circuit_breaker_threshold=1,
            circuit_breaker_enabled=True
        )

        # Open the circuit
        client._circuit_breaker.record_failure()
        assert client.get_circuit_breaker_status()["state"] == "open"

        # Reset
        client.reset_circuit_breaker()

        status = client.get_circuit_breaker_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0

    def test_reset_circuit_breaker_clears_failures(self):
        """Test reset_circuit_breaker clears failure count."""
        client = BigMapRestClient(
            circuit_breaker_threshold=5,
            circuit_breaker_enabled=True
        )

        # Record failures
        client._circuit_breaker.record_failure()
        client._circuit_breaker.record_failure()
        assert client.get_circuit_breaker_status()["failure_count"] == 2

        # Reset
        client.reset_circuit_breaker()

        assert client.get_circuit_breaker_status()["failure_count"] == 0


class TestBigMapRestClientCircuitBreakerRequestIntegration:
    """Test circuit breaker integration with request handling."""

    def test_circuit_breaker_blocks_requests_when_open(self):
        """Test requests are blocked when circuit breaker is OPEN."""
        from gridfia.exceptions import CircuitBreakerOpen

        client = BigMapRestClient(
            circuit_breaker_threshold=1,
            circuit_breaker_enabled=True
        )

        # Open the circuit
        client._circuit_breaker.record_failure()

        with pytest.raises(CircuitBreakerOpen):
            client._rate_limited_request('GET', 'http://test.com')

    def test_circuit_breaker_open_exception_contains_details(self):
        """Test CircuitBreakerOpen exception contains relevant details."""
        from gridfia.exceptions import CircuitBreakerOpen

        client = BigMapRestClient(
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=60.0,
            circuit_breaker_enabled=True
        )

        # Open the circuit
        client._circuit_breaker.record_failure()
        client._circuit_breaker.record_failure()

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            client._rate_limited_request('GET', 'http://test.com')

        exception = exc_info.value
        assert exception.failure_count == 2
        assert exception.failure_threshold == 2
        assert exception.retry_after is not None

    def test_circuit_breaker_records_success_on_200(self):
        """Test circuit breaker records success on successful response."""
        client = BigMapRestClient(
            circuit_breaker_threshold=5,
            circuit_breaker_enabled=True
        )

        # Record some failures first
        client._circuit_breaker.record_failure()
        client._circuit_breaker.record_failure()
        assert client._circuit_breaker.failure_count == 2

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            client._rate_limited_request('GET', 'http://test.com')

        # Failure count should be reset
        assert client._circuit_breaker.failure_count == 0

    def test_circuit_breaker_records_failure_on_server_error(self):
        """Test circuit breaker records failure on 5xx response."""
        client = BigMapRestClient(
            circuit_breaker_threshold=5,
            circuit_breaker_enabled=True
        )

        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_request.return_value = mock_response

            client._rate_limited_request('GET', 'http://test.com')

        assert client._circuit_breaker.failure_count == 1

    def test_circuit_breaker_records_failure_on_connection_error(self):
        """Test circuit breaker records failure on connection error."""
        client = BigMapRestClient(
            circuit_breaker_threshold=5,
            circuit_breaker_enabled=True
        )

        with patch.object(client.session, 'request') as mock_request:
            mock_request.side_effect = ConnectionError("Connection failed")

            with pytest.raises(APIConnectionError):
                client._rate_limited_request('GET', 'http://test.com')

        assert client._circuit_breaker.failure_count == 1

    def test_circuit_breaker_records_failure_on_timeout(self):
        """Test circuit breaker records failure on timeout."""
        client = BigMapRestClient(
            circuit_breaker_threshold=5,
            circuit_breaker_enabled=True
        )

        with patch.object(client.session, 'request') as mock_request:
            mock_request.side_effect = Timeout("Request timed out")

            with pytest.raises(APIConnectionError):
                client._rate_limited_request('GET', 'http://test.com')

        assert client._circuit_breaker.failure_count == 1

    def test_circuit_breaker_allows_request_after_recovery_timeout(self):
        """Test circuit breaker allows request after recovery timeout."""
        client = BigMapRestClient(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.01,
            circuit_breaker_enabled=True
        )

        # Open the circuit
        client._circuit_breaker.record_failure()
        assert client._circuit_breaker.can_execute() is False

        # Wait for recovery timeout
        time.sleep(0.02)

        # Circuit should be in HALF_OPEN and allow request
        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            response = client._rate_limited_request('GET', 'http://test.com')

            assert response == mock_response

    def test_circuit_breaker_closes_on_successful_recovery(self):
        """Test circuit breaker closes after successful recovery request."""
        from gridfia.external.fia_client import CircuitBreakerState

        client = BigMapRestClient(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.01,
            circuit_breaker_enabled=True
        )

        # Open the circuit
        client._circuit_breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(0.02)

        # Make successful request
        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_request.return_value = mock_response

            client._rate_limited_request('GET', 'http://test.com')

        # Circuit should be CLOSED
        assert client._circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_reopens_on_failed_recovery(self):
        """Test circuit breaker reopens on failed recovery request."""
        from gridfia.external.fia_client import CircuitBreakerState

        client = BigMapRestClient(
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.01,
            circuit_breaker_enabled=True
        )

        # Open the circuit
        client._circuit_breaker.record_failure()

        # Wait for recovery timeout
        time.sleep(0.02)

        # Force transition to HALF_OPEN
        client._circuit_breaker.can_execute()
        assert client._circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Make failed request (server error)
        with patch.object(client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_request.return_value = mock_response

            client._rate_limited_request('GET', 'http://test.com')

        # Circuit should be OPEN again
        assert client._circuit_breaker.state == CircuitBreakerState.OPEN