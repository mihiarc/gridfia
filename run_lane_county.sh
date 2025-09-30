#!/bin/bash
# Run Lane County analysis pipeline

set -e  # Exit on error

echo "Lane County Forest Analysis Pipeline"
echo "===================================="

# Activate virtual environment
source .venv/bin/activate

# Step 1: Download data and create Zarr store
echo ""
echo "Step 1: Downloading species data and building Zarr store..."
echo "------------------------------------------------------------"
python3 download_lane_county.py

# Step 2: Analyze and visualize
echo ""
echo "Step 2: Analyzing forest metrics and creating visualizations..."
echo "----------------------------------------------------------------"
python3 analyze_lane_county.py

echo ""
echo "Pipeline complete!"
echo "Results available in:"
echo "  - Zarr store: data/lane_county/lane_county.zarr"
echo "  - Metrics: output/lane_county/metrics/"
echo "  - Maps: output/lane_county/maps/"