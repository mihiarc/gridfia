# Changelog

All notable changes to GridFIA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom polygon boundary support with automatic clipping (#18)
  - `polygon` parameter on `download_species()`
  - `clip_to_polygon` parameter on `create_zarr()`
  - `LocationConfig.from_polygon()` class method
  - Polygon utilities module for loading and clipping
- Contributing guidelines (CONTRIBUTING.md)
- Changelog (CHANGELOG.md)
- Code of Conduct (CODE_OF_CONDUCT.md)
- GitHub issue and PR templates

## [0.5.0] - 2024-12-26

### Added
- Configurable cloud storage backend with Backblaze B2 as default
- State dataset loading API (`load_state()`, `load_from_cloud()`)
- US-wide data processing pipeline scripts
- `list_state_datasets()` and `list_sample_datasets()` API methods

### Changed
- Rebranded from BigMap to GridFIA (Forest Inventory Applications)
- Rewrote documentation for API-first architecture
- Default cloud storage now uses public Backblaze B2 bucket

### Fixed
- Storage options for HTTP URLs (public B2 access)
- Zarr validation to handle empty species names

## [0.4.0] - 2024-12-20

### Added
- Cloud storage streaming support for Zarr stores
- Remote data access from HTTP, S3, R2, and GCS backends
- `ZarrStore` class for unified local/cloud access
- Google Colab tutorial notebook
- `download_sample()` API method for quick-start data

### Changed
- Zarr stores now use v3 API for cloud compatibility
- Improved fsspec integration for remote storage

### Fixed
- Zarr v3 API compatibility in download_sample
- Examples to download all species for valid diversity metrics

## [0.3.2] - 2024-12-15

### Added
- SEO improvements for documentation site
- FIAtools.org ecosystem links

### Changed
- Updated documentation with fiatools.org branding

## [0.3.1] - 2024-12-10

### Changed
- Updated documentation dependencies
- Improved mkdocs configuration

## [0.3.0] - 2024-12-01

### Added
- Initial public release
- Core GridFIA API with single entry point
- FIA BIGMAP REST API integration for species data download
- Zarr store creation from GeoTIFF files
- Forest metrics calculations:
  - Species richness
  - Shannon diversity index
  - Simpson diversity index
  - Pielou's evenness
  - Total biomass
  - Species proportions
- Visualization module with matplotlib-based mapping
- Location configuration system for states, counties, and custom regions
- Comprehensive test suite with 80%+ coverage
- MkDocs documentation site

### Infrastructure
- GitHub Actions for documentation deployment
- PyPI package distribution
- uv-based development environment

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.5.0 | 2024-12-26 | Cloud storage backends, GridFIA rebrand |
| 0.4.0 | 2024-12-20 | Cloud streaming, Colab notebook |
| 0.3.2 | 2024-12-15 | SEO, fiatools.org links |
| 0.3.1 | 2024-12-10 | Documentation updates |
| 0.3.0 | 2024-12-01 | Initial public release |
