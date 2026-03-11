"""
Tests for ZarrStore Icechunk backend integration.

Uses in-memory Icechunk repositories (no credentials, no network).
"""

from typing import Optional

import numpy as np
import pytest
import zarr

try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

from gridfia.utils.zarr_utils import YearSlicedArray, ZarrStore
from gridfia.exceptions import InvalidZarrStructure

pytestmark = pytest.mark.skipif(
    not HAS_ICECHUNK,
    reason="icechunk package not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_repo_4d(
    n_years: int = 1,
    n_species: int = 3,
    height: int = 100,
    width: int = 120,
    years: Optional[list] = None,
    species_codes: Optional[list] = None,
    species_names: Optional[list] = None,
    use_singular_attrs: bool = True,
    use_component_transform: bool = True,
):
    """Create an in-memory Icechunk repo matching the BIGMAP datacube schema."""
    if years is None:
        years = [2018] if n_years == 1 else list(range(2018, 2018 + n_years))
    if species_codes is None:
        species_codes = [f"{i:04d}" for i in range(n_species)]
    if species_names is None:
        species_names = [f"Species {i}" for i in range(n_species)]

    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage=storage)

    session = repo.writable_session("main")
    root = zarr.open_group(store=session.store, mode="w")

    # Create 4D biomass array: (year, species, y, x)
    rng = np.random.default_rng(42)
    data = rng.random((n_years, n_species, height, width)).astype(np.float32) * 100

    root.create_array(
        "biomass",
        data=data,
        chunks=(1, 1, min(64, height), min(64, width)),
    )

    # Metadata: Icechunk datacube conventions
    root.attrs["years"] = years

    if use_singular_attrs:
        # Icechunk datacube uses singular attribute names
        root.attrs["species_code"] = species_codes
        root.attrs["species_name"] = species_names
    else:
        # GridFIA uses plural attribute names
        root.attrs["species_codes"] = species_codes
        root.attrs["species_names"] = species_names

    root.attrs["crs"] = "EPSG:5070"

    if use_component_transform:
        # Icechunk datacube stores transform as components
        root.attrs["transform_origin_x"] = -2361915.0
        root.attrs["transform_origin_y"] = 3177435.0
        root.attrs["resolution_m"] = 30.0
    else:
        # 6-element affine list
        root.attrs["transform"] = [30.0, 0.0, -2361915.0, 0.0, -30.0, 3177435.0]

    session.commit("Initial test data")
    return repo


# ---------------------------------------------------------------------------
# YearSlicedArray tests
# ---------------------------------------------------------------------------

class TestYearSlicedArray:
    """Tests for the YearSlicedArray wrapper."""

    def test_shape_is_3d(self):
        storage = icechunk.in_memory_storage()
        repo = icechunk.Repository.create(storage=storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")
        data = np.zeros((2, 3, 50, 60), dtype=np.float32)
        root.create_array("biomass", data=data, chunks=(1, 1, 50, 60))
        session.commit("test")

        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        arr = root["biomass"]

        wrapper = YearSlicedArray(arr, year_index=0)
        assert wrapper.shape == (3, 50, 60)
        assert wrapper.ndim == 3

    def test_full_slice(self):
        repo = create_test_repo_4d(n_years=1, n_species=2, height=10, width=15)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw = root["biomass"]

        wrapper = YearSlicedArray(raw, year_index=0)
        result = wrapper[:]
        expected = raw[0, :, :, :]

        np.testing.assert_array_equal(result, expected)

    def test_single_species_indexing(self):
        repo = create_test_repo_4d(n_years=1, n_species=3, height=10, width=15)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw = root["biomass"]

        wrapper = YearSlicedArray(raw, year_index=0)
        result = wrapper[1, :, :]
        expected = raw[0, 1, :, :]

        np.testing.assert_array_equal(result, expected)

    def test_integer_index(self):
        repo = create_test_repo_4d(n_years=1, n_species=3, height=10, width=15)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw = root["biomass"]

        wrapper = YearSlicedArray(raw, year_index=0)
        result = wrapper[2]
        expected = raw[0, 2, :, :]

        np.testing.assert_array_equal(result, expected)

    def test_spatial_chunk_indexing(self):
        repo = create_test_repo_4d(n_years=1, n_species=3, height=50, width=60)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw = root["biomass"]

        wrapper = YearSlicedArray(raw, year_index=0)
        result = wrapper[:, 10:20, 30:40]
        expected = raw[0, :, 10:20, 30:40]

        np.testing.assert_array_equal(result, expected)

    def test_multi_year_selects_correct_year(self):
        repo = create_test_repo_4d(
            n_years=3, n_species=2, height=10, width=10,
            years=[2016, 2018, 2020],
        )
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw = root["biomass"]

        # Select year index 1 (2018)
        wrapper = YearSlicedArray(raw, year_index=1)
        result = wrapper[:]
        expected = raw[1, :, :, :]

        np.testing.assert_array_equal(result, expected)
        assert wrapper.shape == (2, 10, 10)

    def test_invalid_ndim_raises(self):
        storage = icechunk.in_memory_storage()
        repo = icechunk.Repository.create(storage=storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")
        root.create_array("data", data=np.zeros((10, 20)), chunks=(10, 20))
        session.commit("test")

        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")

        with pytest.raises(ValueError, match="Expected 4D"):
            YearSlicedArray(root["data"], year_index=0)

    def test_out_of_range_year_raises(self):
        repo = create_test_repo_4d(n_years=2, n_species=2, height=5, width=5)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")

        with pytest.raises(IndexError, match="out of range"):
            YearSlicedArray(root["biomass"], year_index=5)

    def test_dtype(self):
        repo = create_test_repo_4d(n_species=2, height=5, width=5)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")

        wrapper = YearSlicedArray(root["biomass"], year_index=0)
        assert wrapper.dtype == np.float32

    def test_len(self):
        repo = create_test_repo_4d(n_species=4, height=5, width=5)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")

        wrapper = YearSlicedArray(root["biomass"], year_index=0)
        assert len(wrapper) == 4

    def test_repr(self):
        repo = create_test_repo_4d(n_species=3, height=10, width=15)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")

        wrapper = YearSlicedArray(root["biomass"], year_index=0)
        repr_str = repr(wrapper)
        assert "YearSlicedArray" in repr_str
        assert "year_index=0" in repr_str


# ---------------------------------------------------------------------------
# ZarrStore.from_icechunk() tests
# ---------------------------------------------------------------------------

class TestZarrStoreFromIcechunk:
    """Tests for ZarrStore.from_icechunk() classmethod."""

    def test_basic_open(self):
        repo = create_test_repo_4d(n_species=3, height=20, width=25)
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store.shape == (3, 20, 25)
        assert store.num_species == 3
        assert len(store.species_codes) == 3
        assert len(store.species_names) == 3

    def test_species_codes_from_singular_attrs(self):
        """Icechunk datacube uses singular 'species_code' attribute."""
        codes = ["0000", "0131", "0202"]
        names = ["Total", "Loblolly Pine", "Douglas-fir"]
        repo = create_test_repo_4d(
            n_species=3, height=10, width=10,
            species_codes=codes,
            species_names=names,
            use_singular_attrs=True,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store.species_codes == codes
        assert store.species_names == names

    def test_species_codes_from_plural_attrs(self):
        """GridFIA convention uses plural 'species_codes'."""
        codes = ["0000", "0131"]
        names = ["Total", "Loblolly Pine"]
        repo = create_test_repo_4d(
            n_species=2, height=10, width=10,
            species_codes=codes,
            species_names=names,
            use_singular_attrs=False,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store.species_codes == codes
        assert store.species_names == names

    def test_component_transform_reconstruction(self):
        """Icechunk stores transform as origin_x, origin_y, resolution_m."""
        repo = create_test_repo_4d(
            n_species=2, height=10, width=10,
            use_component_transform=True,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        transform = store.transform
        assert transform.a == 30.0        # x resolution
        assert transform.e == -30.0       # y resolution (negative = north-up)
        assert transform.c == -2361915.0  # origin x
        assert transform.f == 3177435.0   # origin y

    def test_list_transform(self):
        """Standard 6-element transform list."""
        repo = create_test_repo_4d(
            n_species=2, height=10, width=10,
            use_component_transform=False,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        transform = store.transform
        assert transform.a == 30.0
        assert transform.c == -2361915.0

    def test_crs(self):
        repo = create_test_repo_4d(n_species=2, height=10, width=10)
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert str(store.crs) == "EPSG:5070"

    def test_biomass_data_access(self):
        """Verify data reads correctly through the 3D wrapper."""
        repo = create_test_repo_4d(n_species=2, height=10, width=15)

        # Get raw data for comparison
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw_year0 = root["biomass"][0, :, :, :]

        store = ZarrStore.from_icechunk(repo, year=2018)
        result = store.biomass[:]

        np.testing.assert_array_equal(result, raw_year0)

    def test_get_species_layer(self):
        codes = ["0000", "0131"]
        repo = create_test_repo_4d(
            n_species=2, height=10, width=15,
            species_codes=codes,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        layer = store.get_species_layer("0131")
        assert layer.shape == (10, 15)

    def test_get_species_index(self):
        codes = ["0000", "0131", "0202"]
        repo = create_test_repo_4d(
            n_species=3, height=10, width=10,
            species_codes=codes,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store.get_species_index("0131") == 1
        assert store.get_species_index("0202") == 2

    def test_multi_year_selects_correct_data(self):
        repo = create_test_repo_4d(
            n_years=3, n_species=2, height=10, width=10,
            years=[2016, 2018, 2020],
        )

        # Get raw 2018 data (index 1)
        session = repo.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        raw_2018 = root["biomass"][1, :, :, :]

        store = ZarrStore.from_icechunk(repo, year=2018)
        result = store.biomass[:]

        np.testing.assert_array_equal(result, raw_2018)

    def test_invalid_year_raises(self):
        repo = create_test_repo_4d(
            n_years=1, n_species=2, height=10, width=10,
            years=[2018],
        )

        with pytest.raises(ValueError, match="Year 2020 not found"):
            ZarrStore.from_icechunk(repo, year=2020)

    def test_missing_biomass_raises(self):
        storage = icechunk.in_memory_storage()
        repo = icechunk.Repository.create(storage=storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")
        root.attrs["test"] = True
        session.commit("empty repo")

        with pytest.raises(InvalidZarrStructure, match="missing required 'biomass'"):
            ZarrStore.from_icechunk(repo)

    def test_3d_biomass_raises(self):
        """from_icechunk expects 4D, not 3D."""
        storage = icechunk.in_memory_storage()
        repo = icechunk.Repository.create(storage=storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")
        root.create_array(
            "biomass",
            data=np.zeros((3, 10, 10), dtype=np.float32),
            chunks=(1, 10, 10),
        )
        session.commit("3D data")

        with pytest.raises(InvalidZarrStructure, match="Expected 4D"):
            ZarrStore.from_icechunk(repo)

    def test_no_years_attr_defaults_to_index_zero(self):
        """When no 'years' attr exists, defaults to year_index=0."""
        storage = icechunk.in_memory_storage()
        repo = icechunk.Repository.create(storage=storage)
        session = repo.writable_session("main")
        root = zarr.open_group(store=session.store, mode="w")
        root.create_array(
            "biomass",
            data=np.ones((1, 2, 10, 10), dtype=np.float32),
            chunks=(1, 1, 10, 10),
        )
        root.attrs["species_code"] = ["0000", "0131"]
        root.attrs["species_name"] = ["Total", "Loblolly"]
        session.commit("no years attr")

        store = ZarrStore.from_icechunk(repo, year=2018)
        assert store.shape == (2, 10, 10)

    def test_get_species_info(self):
        codes = ["0000", "0131", "0202"]
        names = ["Total", "Loblolly Pine", "Douglas-fir"]
        repo = create_test_repo_4d(
            n_species=3, height=10, width=10,
            species_codes=codes,
            species_names=names,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        info = store.get_species_info()
        assert len(info) == 3
        assert info[1]["code"] == "0131"
        assert info[1]["name"] == "Loblolly Pine"

    def test_summary(self):
        repo = create_test_repo_4d(n_species=2, height=10, width=15)
        store = ZarrStore.from_icechunk(repo, year=2018)

        summary = store.summary()
        assert summary["shape"] == (2, 10, 15)
        assert summary["path"] is None  # No local path for icechunk
        assert summary["num_species"] == 2
        assert summary["crs"] == "EPSG:5070"

    def test_repr_shows_icechunk(self):
        repo = create_test_repo_4d(n_species=2, height=10, width=10)
        store = ZarrStore.from_icechunk(repo, year=2018)

        repr_str = repr(store)
        assert "icechunk" in repr_str
        assert "year=2018" in repr_str

    def test_bounds_from_transform_and_shape(self):
        """When no bounds attr, reconstruct from transform + shape."""
        repo = create_test_repo_4d(
            n_species=2, height=100, width=200,
            use_component_transform=True,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        bounds = store.bounds
        # origin_x = -2361915, resolution = 30, width = 200
        # left = -2361915, right = -2361915 + 200*30 = -2355915
        # origin_y = 3177435, resolution = 30, height = 100
        # top = 3177435, bottom = 3177435 + 100*(-30) = 3174435
        assert bounds[0] == -2361915.0  # left
        assert bounds[3] == 3177435.0   # top

    def test_height_width_properties(self):
        repo = create_test_repo_4d(n_species=2, height=30, width=45)
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store.height == 30
        assert store.width == 45

    def test_dtype(self):
        repo = create_test_repo_4d(n_species=2, height=10, width=10)
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store.dtype == np.float32

    def test_get_extent(self):
        repo = create_test_repo_4d(
            n_species=2, height=100, width=200,
            use_component_transform=True,
        )
        store = ZarrStore.from_icechunk(repo, year=2018)

        extent = store.get_extent()
        # (left, right, bottom, top)
        assert len(extent) == 4
        left, right, bottom, top = extent
        assert left == -2361915.0
        assert right == -2361915.0 + 200 * 30.0
        assert top == 3177435.0
        assert bottom == 3177435.0 + 100 * (-30.0)

    def test_close_cleans_up_icechunk_resources(self):
        """close() should release Icechunk session and repo references."""
        repo = create_test_repo_4d(n_species=2, height=10, width=10)
        store = ZarrStore.from_icechunk(repo, year=2018)

        assert store._session is not None
        assert store._repo is not None

        store.close()

        assert store._closed is True
        assert store._session is None
        assert store._repo is None
