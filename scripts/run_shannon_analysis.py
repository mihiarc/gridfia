#!/usr/bin/env python3
"""
Comprehensive North Carolina Shannon Diversity Analysis
Using BigMap's ForestMetricsProcessor with 327 species
"""

import sys
from pathlib import Path
from bigmap.core.processors.forest_metrics import run_forest_analysis

def main():
    print('🌲 Starting Comprehensive NC Shannon Diversity Analysis')
    print('=' * 60)
    print(f'📁 Data: output/nc_biomass_expandable.zarr')
    print(f'⚙️  Config: nc_comprehensive_shannon_config.yaml')
    print(f'🎯 Target: All 326 species (excluding TOTAL layer)')
    print(f'📊 Output: Shannon Diversity + context metrics')
    print('')
    
    try:
        # Run the analysis
        results = run_forest_analysis(
            zarr_path='output/nc_biomass_expandable.zarr',
            config_path='nc_comprehensive_shannon_config.yaml'
        )
        
        print(f'\n✅ Analysis Complete! Generated {len(results)} outputs:')
        print('-' * 50)
        for name, path in results.items():
            file_size = Path(path).stat().st_size / (1024*1024)  # MB
            print(f'  📊 {name:35} -> {path} ({file_size:.1f} MB)')
        
        print(f'\n🎉 Success! All outputs saved to: output/nc_comprehensive_shannon/')
        return results
        
    except Exception as e:
        print(f'❌ Error during analysis: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    results = main() 