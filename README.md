# Heirs Property Analysis System

A spatial data analysis system for tracking and analyzing heirs property parcels in North Carolina.

## Project Status

- Phase 1 (Infrastructure): ✅ Complete
- Phase 2 (Data Pipeline): 🔄 In Progress
- Phase 3 (Analysis Tools): 📅 Planned

## Features

- PostGIS spatial database with NC State Plane support
- JupyterLab environment for data analysis
- Automated data processing pipeline
- Comprehensive testing framework
- Docker-based deployment

## Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Make (optional, for using Makefile commands)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heirs-property.git
   cd heirs-property
   ```

2. Create environment file:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access JupyterLab:
   - Open http://localhost:8888 in your browser
   - Token is printed in the jupyter container logs

## Development Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   python -m pytest tests/
   ```

3. Start development environment:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

## Project Structure

```
heirs-property/
├── data/               # Data directory
│   ├── raw/           # Raw input data
│   ├── processed/     # Processed data
│   └── interim/       # Intermediate data
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── processing/   # Data processing scripts
│   └── analysis/     # Analysis tools
├── tests/            # Test suite
└── docker/           # Docker configuration
```

## Testing

- Integration tests: `pytest -m integration`
- Unit tests: `pytest -m "not integration"`
- All tests: `pytest`

## Documentation

- [Phase 1 Progress](docs/phase1_progress.md)
- [Testing Plan](docs/testing_plan.md)
- [Database Setup](docs/database_setup.md)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests and ensure they pass
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PostGIS team for spatial database support
- JupyterLab team for the analysis environment
- Contributors and maintainers