#!/usr/bin/env python3

import rasterio
import numpy as np
from pathlib import Path
import json
import sys

def get_raster_info(filepath):
    """Extract metadata and basic statistics from a raster file."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        valid_data = data[data != src.nodata] if src.nodata is not None else data
        return {
            'filename': filepath.name,
            'crs': src.crs.to_string(),
            'width': src.width,
            'height': src.height,
            'resolution': src.res,
            'bounds': {
                'left': src.bounds.left,
                'bottom': src.bounds.bottom,
                'right': src.bounds.right,
                'top': src.bounds.top
            },
            'nodata': float(src.nodata) if src.nodata is not None else None,
            'dtype': str(src.dtypes[0]),
            'stats': {
                'min': float(np.nanmin(valid_data)),
                'max': float(np.nanmax(valid_data)),
                'mean': float(np.nanmean(valid_data)),
                'std': float(np.nanstd(valid_data)),
                'missing_count': int(np.sum(data == src.nodata)) if src.nodata is not None else 0,
                'valid_count': int(np.sum(data != src.nodata)) if src.nodata is not None else len(data.flatten())
            }
        }

def main():
    """Analyze NDVI raster files and output results."""
    data_dir = Path('data/raw/ndvi_Heirs-Vance_County')
    ndvi_files = sorted(list(data_dir.glob('*.tif')))
    
    if not ndvi_files:
        print(f"No NDVI files found in: {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(ndvi_files)} NDVI files:")
    for f in ndvi_files:
        print(f"- {f.name}")
    
    print("\nAnalyzing raster characteristics...")
    results = []
    for f in ndvi_files:
        print(f"\nAnalyzing {f.name}...")
        try:
            info = get_raster_info(f)
            results.append(info)
            print(f"Successfully analyzed {f.name}")
        except Exception as e:
            print(f"Error analyzing {f.name}: {str(e)}")
    
    output_file = Path('data/processed/ndvi_analysis_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\nSummary:")
    for result in results:
        print(f"\n{result['filename']}:")
        print(f"  Resolution: {result['resolution']} degrees")
        print(f"  Size: {result['width']} x {result['height']} pixels")
        print(f"  Valid pixels: {result['stats']['valid_count']:,}")
        print(f"  Missing pixels: {result['stats']['missing_count']:,}")
        print(f"  NDVI range: {result['stats']['min']:.4f} to {result['stats']['max']:.4f}")
        print(f"  Mean NDVI: {result['stats']['mean']:.4f} ± {result['stats']['std']:.4f}")

if __name__ == '__main__':
    main() 