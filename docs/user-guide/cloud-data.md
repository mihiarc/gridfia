# Cloud Data Access

GridFIA hosts pre-processed forest data on cloud storage, enabling streaming access without downloading from the FIA API. This is the fastest way to start working with forest data.

## Quick Start

```python
from gridfia import GridFIA

api = GridFIA()

# Load a sample dataset (streaming -- only fetches chunks you access)
store = api.load_from_cloud(sample="durham_nc")
print(f"Shape: {store.shape}")
print(f"Species: {store.num_species}")

# Access data -- only the requested chunks are downloaded
biomass_subset = store.biomass[:, 100:200, 100:200]
```

## Available Sample Datasets

List sample datasets programmatically:

```python
api = GridFIA()
samples = api.list_sample_datasets()
for s in samples:
    print(f"{s['key']}: {s['name']} ({s['approximate_size_mb']} MB)")
```

| Sample Key | Description | Species | Size |
|-----------|-------------|---------|------|
| `durham_nc` | Durham County, North Carolina | 326 | ~263 MB |

## Available State Datasets

Full-state datasets are available for streaming access:

```python
api = GridFIA()
states = api.list_state_datasets()
for s in states:
    print(f"{s['state']}: {s['name']} ({s['approximate_size_mb']} MB)")
```

| State | Name | Species | Shape | Size |
|-------|------|---------|-------|------|
| `RI` | Rhode Island | 326 | 326 x 3407 x 2264 | ~646 MB |
| `CT` | Connecticut | 326 | 326 x 4100 x 7151 | ~2807 MB |

## Loading Methods

### `load_from_cloud()` -- Load by Sample Name or URL

```python
api = GridFIA()

# Load a pre-hosted sample by name
store = api.load_from_cloud(sample="durham_nc")

# Load from any URL (S3, B2, R2, HTTP)
store = api.load_from_cloud(
    url="https://your-bucket.s3.amazonaws.com/forest_data.zarr"
)

# With custom storage options (e.g., for private buckets)
store = api.load_from_cloud(
    url="s3://my-bucket/forest.zarr",
    storage_options={"key": "...", "secret": "..."}
)
```

### `load_state()` -- Load Pre-Hosted State Data

```python
api = GridFIA()

# Load Rhode Island (streaming)
store = api.load_state("RI")
print(f"Shape: {store.shape}")
print(f"Species: {store.num_species}")

# Access specific species
idx = store.species_codes.index("0316")
red_maple = store.biomass[idx]
```

### `download_sample()` -- Download to Local Storage

For repeated access or large calculations, download to local storage first:

```python
api = GridFIA()

# Download sample to local Zarr store
local_path = api.download_sample("durham_nc", output_path="data/durham.zarr")

# Use locally for faster access
results = api.calculate_metrics(local_path, calculations=["shannon_diversity"])
```

## Streaming vs Download

| Approach | Best For | Trade-offs |
|----------|----------|------------|
| **Streaming** (`load_from_cloud`, `load_state`) | Exploratory analysis, accessing subsets | Slower for full-dataset calculations, requires internet |
| **Download** (`download_sample`) | Repeated analysis, large calculations | One-time download cost, faster subsequent access |

**Recommendation**: Use streaming for exploring data and accessing subsets. Download locally when you need to run calculations on the full dataset.

```python
api = GridFIA()

# Streaming: great for exploration
store = api.load_from_cloud(sample="durham_nc")
subset = store.biomass[:, 100:200, 100:200]  # Only downloads this chunk

# Local: better for full calculations
local_path = api.download_sample("durham_nc", output_path="data/durham.zarr")
results = api.calculate_metrics(local_path)
```

## Cloud URL Patterns

GridFIA supports multiple cloud storage backends:

| Backend | URL Pattern |
|---------|-------------|
| Backblaze B2 | `https://f004.backblazeb2.com/file/bucket-name/path.zarr` |
| Cloudflare R2 | `https://account.r2.dev/path.zarr` |
| AWS S3 | `s3://bucket-name/path.zarr` or `https://bucket.s3.amazonaws.com/path.zarr` |
| HTTP/HTTPS | Any valid URL pointing to a Zarr store |

## Configuring Cloud Storage

See [Configuration](configuration.md#cloud-storage-configuration) for details on configuring custom cloud backends via `CloudStorageConfig`.

## See Also

- [Configuration](configuration.md) - Cloud storage configuration
- [Data Pipeline](data-pipeline.md) - Downloading from the FIA API directly
- [API Reference: GridFIA](../api/gridfia.md) - Full method documentation
