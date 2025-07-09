#!/usr/bin/env python3
"""
Analyze species composition at HEIRS property locations.

This script:
1. Loads parcel data (PARCEL_LID + geometry only)
2. Reprojects coordinates from WGS84 to zarr CRS (ESRI:102039)
3. Extracts species biomass data from zarr at parcel locations
4. Calculates unique species and their relative proportions
5. Outputs results with species composition analysis
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import zarr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import Point

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeirsSpeciesAnalyzer:
    """Analyzer for species composition at HEIRS property locations."""
    
    def __init__(self, zarr_path: str = "output/nc_biomass_expandable.zarr"):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        zarr_path : str
            Path to the species biomass zarr container
        """
        self.zarr_path = zarr_path
        self.zarr_array = None
        self.zarr_metadata = {}
        self.species_codes = []
        self.species_names = []
        
        # Load zarr data
        self._load_zarr_data()
        
    def _load_zarr_data(self):
        """Load and validate zarr array."""
        console.print(f"\n[bold blue]🗂️  Loading zarr data from: {self.zarr_path}[/bold blue]")
        
        if not Path(self.zarr_path).exists():
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")
        
        # Open zarr array
        self.zarr_array = zarr.open_array(self.zarr_path, mode='r')
        
        # Extract metadata
        self.species_codes = self.zarr_array.attrs.get('species_codes', [])
        self.species_names = self.zarr_array.attrs.get('species_names', [])
        
        # Extract spatial metadata
        self.zarr_metadata = {
            'crs': self.zarr_array.attrs.get('crs', 'ESRI:102039'),
            'bounds': self.zarr_array.attrs.get('bounds'),
            'transform': self.zarr_array.attrs.get('transform'),
            'height': self.zarr_array.attrs.get('height', self.zarr_array.shape[1]),
            'width': self.zarr_array.attrs.get('width', self.zarr_array.shape[2])
        }
        
        console.print(f"✅ Zarr loaded: {self.zarr_array.shape}")
        console.print(f"📊 Species count: {len(self.species_codes)}")
        console.print(f"🗺️  CRS: {self.zarr_metadata['crs']}")
        
    def load_parcel_data(self, parquet_path: str) -> gpd.GeoDataFrame:
        """
        Load parcel data with only essential columns.
        
        Parameters
        ----------
        parquet_path : str
            Path to parquet file with parcel data
            
        Returns
        -------
        gpd.GeoDataFrame
            Simplified parcel data with PARCEL_LID and geometry
        """
        console.print(f"\n[bold green]📍 Loading parcel data from: {parquet_path}[/bold green]")
        
        # Load only essential columns
        gdf = gpd.read_parquet(parquet_path)
        
        # Check if required columns exist
        required_cols = ['PARCEL_LID', 'geometry']
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Keep only essential columns
        gdf_simple = gdf[required_cols].copy()
        
        # Remove parcels with missing data
        initial_count = len(gdf_simple)
        gdf_simple = gdf_simple.dropna()
        final_count = len(gdf_simple)
        
        console.print(f"📦 Loaded {final_count:,} parcels (removed {initial_count - final_count:,} with missing data)")
        console.print(f"🗺️  Original CRS: {gdf_simple.crs}")
        
        # Reproject to zarr CRS if needed
        zarr_crs = self.zarr_metadata['crs']
        if str(gdf_simple.crs) != zarr_crs:
            console.print(f"🔄 Reprojecting to zarr CRS: {zarr_crs}")
            gdf_simple = gdf_simple.to_crs(zarr_crs)
        
        return gdf_simple
        
    def extract_species_at_parcels(self, parcels: gpd.GeoDataFrame, 
                                   biomass_threshold: float = 0.1) -> pd.DataFrame:
        """
        Extract species biomass data at parcel locations.
        
        Parameters
        ----------
        parcels : gpd.GeoDataFrame
            Parcel data with geometry
        biomass_threshold : float
            Minimum biomass to consider species as present
            
        Returns
        -------
        pd.DataFrame
            Species composition data for each parcel
        """
        console.print(f"\n[bold yellow]🔍 Extracting species data at {len(parcels):,} parcel locations[/bold yellow]")
        
        results = []
        
        # Get zarr transform for coordinate conversion
        if self.zarr_metadata['transform']:
            transform = rasterio.transform.Affine(*self.zarr_metadata['transform'])
        else:
            # Compute transform from bounds
            bounds = self.zarr_metadata['bounds']
            height = self.zarr_metadata['height']
            width = self.zarr_metadata['width']
            transform = from_bounds(*bounds, width, height)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Extracting species data...", total=len(parcels))
            
            for idx, parcel in parcels.iterrows():
                parcel_id = parcel['PARCEL_LID']
                geometry = parcel['geometry']
                
                try:
                    # Get centroid for point extraction
                    if hasattr(geometry, 'centroid'):
                        point = geometry.centroid
                    else:
                        point = geometry
                    
                    # Convert coordinates to pixel indices
                    col, row = ~transform * (point.x, point.y)
                    col, row = int(col), int(row)
                    
                    # Check if point is within zarr bounds
                    if (0 <= row < self.zarr_array.shape[1] and 
                        0 <= col < self.zarr_array.shape[2]):
                        
                        # Extract species biomass at this location
                        species_biomass = self.zarr_array[:, row, col]
                        
                        # Calculate species composition
                        present_species = species_biomass >= biomass_threshold
                        total_biomass = np.sum(species_biomass[present_species])
                        
                        if total_biomass > 0:
                            # Calculate proportions
                            proportions = species_biomass[present_species] / total_biomass
                            
                            # Get species info for present species
                            present_indices = np.where(present_species)[0]
                            
                            result = {
                                'PARCEL_LID': parcel_id,
                                'total_biomass': total_biomass,
                                'species_count': len(present_indices),
                                'x_coord': point.x,
                                'y_coord': point.y,
                                'pixel_row': row,
                                'pixel_col': col
                            }
                            
                            # Add species-specific data
                            for i, species_idx in enumerate(present_indices):
                                species_code = self.species_codes[species_idx] if species_idx < len(self.species_codes) else f"SP_{species_idx}"
                                species_name = self.species_names[species_idx] if species_idx < len(self.species_names) else f"Species {species_idx}"
                                
                                result[f'species_{i+1}_code'] = species_code
                                result[f'species_{i+1}_name'] = species_name
                                result[f'species_{i+1}_biomass'] = species_biomass[species_idx]
                                result[f'species_{i+1}_proportion'] = proportions[i]
                            
                            results.append(result)
                        else:
                            # No biomass at this location
                            results.append({
                                'PARCEL_LID': parcel_id,
                                'total_biomass': 0,
                                'species_count': 0,
                                'x_coord': point.x,
                                'y_coord': point.y,
                                'pixel_row': row,
                                'pixel_col': col
                            })
                    else:
                        # Outside zarr bounds
                        results.append({
                            'PARCEL_LID': parcel_id,
                            'total_biomass': np.nan,
                            'species_count': 0,
                            'x_coord': point.x,
                            'y_coord': point.y,
                            'pixel_row': -1,
                            'pixel_col': -1,
                            'note': 'Outside zarr coverage area'
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing parcel {parcel_id}: {e}")
                    results.append({
                        'PARCEL_LID': parcel_id,
                        'total_biomass': np.nan,
                        'species_count': 0,
                        'note': f'Processing error: {str(e)}'
                    })
                
                progress.advance(task)
        
        results_df = pd.DataFrame(results)
        console.print(f"✅ Extracted data for {len(results_df):,} parcels")
        
        return results_df
    
    def analyze_species_composition(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze overall species composition across all parcels.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results from extract_species_at_parcels
            
        Returns
        -------
        Dict
            Summary statistics and species composition analysis
        """
        console.print(f"\n[bold magenta]📈 Analyzing species composition across parcels[/bold magenta]")
        
        # Filter to parcels with biomass data
        valid_parcels = results_df[results_df['total_biomass'] > 0].copy()
        
        if len(valid_parcels) == 0:
            console.print("[red]❌ No parcels with valid biomass data found[/red]")
            return {}
        
        # Collect all species across parcels
        species_summary = {}
        
        for _, parcel in valid_parcels.iterrows():
            species_count = parcel['species_count']
            
            for i in range(1, species_count + 1):
                code_col = f'species_{i}_code'
                name_col = f'species_{i}_name'
                biomass_col = f'species_{i}_biomass'
                prop_col = f'species_{i}_proportion'
                
                if code_col in parcel and pd.notna(parcel[code_col]):
                    species_code = parcel[code_col]
                    species_name = parcel.get(name_col, species_code)
                    biomass = parcel.get(biomass_col, 0)
                    proportion = parcel.get(prop_col, 0)
                    
                    if species_code not in species_summary:
                        species_summary[species_code] = {
                            'name': species_name,
                            'parcel_count': 0,
                            'total_biomass': 0,
                            'proportions': []
                        }
                    
                    species_summary[species_code]['parcel_count'] += 1
                    species_summary[species_code]['total_biomass'] += biomass
                    species_summary[species_code]['proportions'].append(proportion)
        
        # Calculate summary statistics
        analysis = {
            'total_parcels_analyzed': len(results_df),
            'parcels_with_biomass': len(valid_parcels),
            'parcels_outside_coverage': len(results_df[results_df['total_biomass'].isna()]),
            'unique_species_found': len(species_summary),
            'species_details': {}
        }
        
        # Process each species
        for species_code, data in species_summary.items():
            proportions = np.array(data['proportions'])
            
            analysis['species_details'][species_code] = {
                'name': data['name'],
                'parcel_count': data['parcel_count'],
                'occurrence_rate': data['parcel_count'] / len(valid_parcels),
                'total_biomass': data['total_biomass'],
                'mean_proportion': np.mean(proportions),
                'std_proportion': np.std(proportions),
                'min_proportion': np.min(proportions),
                'max_proportion': np.max(proportions)
            }
        
        return analysis
    
    def display_results(self, analysis: Dict):
        """Display analysis results in formatted tables."""
        
        # Overall summary
        summary_panel = Panel(
            "[bold green]📊 HEIRS Species Composition Analysis[/bold green]\n\n"
            f"Total parcels analyzed: {analysis['total_parcels_analyzed']:,}\n"
            f"Parcels with biomass data: {analysis['parcels_with_biomass']:,}\n"
            f"Parcels outside coverage: {analysis['parcels_outside_coverage']:,}\n"
            f"Unique species found: {analysis['unique_species_found']:,}",
            title="Summary Statistics",
            border_style="green"
        )
        console.print(summary_panel)
        
        # Species composition table
        if analysis['species_details']:
            species_table = Table(title=f"Species Composition at HEIRS Properties")
            species_table.add_column("Species Code", style="cyan")
            species_table.add_column("Species Name", style="green", max_width=30)
            species_table.add_column("Parcel Count", justify="right", style="yellow")
            species_table.add_column("Occurrence %", justify="right", style="blue")
            species_table.add_column("Mean Prop %", justify="right", style="magenta")
            species_table.add_column("Total Biomass", justify="right", style="red")
            
            # Sort by occurrence rate
            sorted_species = sorted(
                analysis['species_details'].items(),
                key=lambda x: x[1]['occurrence_rate'],
                reverse=True
            )
            
            for species_code, data in sorted_species:
                species_table.add_row(
                    species_code,
                    data['name'],
                    f"{data['parcel_count']:,}",
                    f"{data['occurrence_rate']*100:.1f}%",
                    f"{data['mean_proportion']*100:.1f}%",
                    f"{data['total_biomass']:.1f}"
                )
            
            console.print(species_table)
    
    def save_results(self, results_df: pd.DataFrame, analysis: Dict, 
                     output_dir: str = "output"):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed parcel results
        parcel_output = output_path / "heirs_parcel_species_composition.csv"
        results_df.to_csv(parcel_output, index=False)
        console.print(f"💾 Saved parcel results: {parcel_output}")
        
        # Save species summary
        if analysis.get('species_details'):
            species_df = pd.DataFrame.from_dict(analysis['species_details'], orient='index')
            species_output = output_path / "heirs_species_summary.csv"
            species_df.to_csv(species_output)
            console.print(f"💾 Saved species summary: {species_output}")
        
        # Save analysis metadata
        import json
        metadata_output = output_path / "heirs_analysis_metadata.json"
        with open(metadata_output, 'w') as f:
            # Convert numpy types for JSON serialization
            clean_analysis = {
                'total_parcels_analyzed': int(analysis['total_parcels_analyzed']),
                'parcels_with_biomass': int(analysis['parcels_with_biomass']),
                'parcels_outside_coverage': int(analysis['parcels_outside_coverage']),
                'unique_species_found': int(analysis['unique_species_found'])
            }
            json.dump(clean_analysis, f, indent=2)
        console.print(f"💾 Saved metadata: {metadata_output}")

def main():
    """Main analysis function."""
    
    # Configuration
    parquet_path = "data/heirs_with_species_diversity.parquet"
    zarr_path = "output/nc_biomass_expandable.zarr"
    biomass_threshold = 0.1  # Mg/ha
    
    # Header
    header = Panel(
        "[bold]HEIRS Property Species Composition Analysis[/bold]\n\n"
        "Analyzing species composition at HEIRS property locations using:\n"
        f"• Parcel data: {parquet_path}\n"
        f"• Species zarr: {zarr_path}\n"
        f"• Biomass threshold: {biomass_threshold} Mg/ha",
        title="🌲 BigMap HEIRS Analysis",
        border_style="blue"
    )
    console.print(header)
    
    try:
        # Initialize analyzer
        analyzer = HeirsSpeciesAnalyzer(zarr_path)
        
        # Load parcel data
        parcels = analyzer.load_parcel_data(parquet_path)
        
        # Extract species data
        results = analyzer.extract_species_at_parcels(parcels, biomass_threshold)
        
        # Analyze composition
        analysis = analyzer.analyze_species_composition(results)
        
        # Display results
        analyzer.display_results(analysis)
        
        # Save results
        analyzer.save_results(results, analysis)
        
        # Next steps
        console.print(f"\n[bold cyan]🎯 Next Steps:[/bold cyan]")
        console.print("• Review the species composition results in output/")
        console.print("• Use parcel-level data to identify diversity hotspots")
        console.print("• Compare with existing species_diversity values")
        console.print("• Analyze spatial patterns of species distribution")
        
        return results, analysis
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main() 