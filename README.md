# Heirs Property Analysis - North Carolina

## Project Overview
This project analyzes heirs property parcels in North Carolina, focusing on land management patterns and forest health indicators. Heirs' properties are lands that have been passed down informally from generation to generation, often resulting in fractioned ownership that can impact land management decisions.

## Project Objectives
- Identify and analyze heirs property parcels in North Carolina
- Compare forest management practices between heirs and non-heirs properties
- Assess forest health indicators across different property types
- Analyze land use patterns and management decisions

## Project Structure
```
/heirs-property
├── src/
│   ├── data/
│   │   ├── raw/           # Original, immutable data
│   │   ├── processed/     # Cleaned, transformed data
│   │   └── interim/       # Intermediate data
│   ├── models/            # Data models and database schemas
│   ├── processing/        # Data processing scripts
│   ├── analysis/          # Analysis notebooks and scripts
│   └── visualization/     # Visualization scripts
├── tests/                 # Test files
├── docs/                  # Documentation
├── notebooks/            # Jupyter notebooks for exploration
├── config/               # Configuration files
└── results/              # Output files, figures, and reports
```

## Data Pipeline
See [Data Pipeline Documentation](docs/data_pipeline.md) for detailed information about:
- Data processing stages
- Analysis workflows
- Visualization pipelines
- Validation procedures

## Setup and Installation
1. Ensure you have Python 3.9+ installed
2. Install Rye for dependency management:
   ```bash
   curl -sSf https://rye-up.com/get | bash
   ```
3. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/yourusername/heirs-property.git
   cd heirs-property
   rye sync
   ```

## Data Sources
- North Carolina parcel data (NC.gdb)
- Heirs property deliverables (HP_Deliverables.gdb)
- Forest Inventory Analysis (FIA) plot data
- NAIP imagery for NDVI analysis
- Tree canopy cover data

## Contributing
[To be added: Contributing guidelines]

## License
[To be added: License information]