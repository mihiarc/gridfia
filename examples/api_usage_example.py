#!/usr/bin/env python3
"""
BigMap API Usage Examples

This script demonstrates the clean API-first architecture of BigMap,
showing how to use the Python API for all forest analysis tasks.
"""

from pathlib import Path
from bigmap import BigMapAPI


def example_basic_workflow():
    """Basic workflow: download, process, analyze, visualize."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Workflow")
    print("="*60)
    
    # Initialize API
    api = BigMapAPI()
    
    # 1. List available species
    print("\n1. Listing available species...")
    species = api.list_species()
    print(f"   Found {len(species)} species")
    print(f"   First 3: {[s.common_name for s in species[:3]]}")
    
    # 2. Download species data for a small area
    print("\n2. Downloading species data for Wake County, NC...")
    files = api.download_species(
        state="North Carolina",
        county="Wake",
        species_codes=["0131", "0068"],  # Loblolly Pine, Baldcypress
        output_dir="data/wake_county"
    )
    print(f"   Downloaded {len(files)} files")
    
    # 3. Create Zarr store
    print("\n3. Creating Zarr store from downloaded data...")
    zarr_path = api.create_zarr(
        input_dir="data/wake_county",
        output_path="data/wake_county.zarr",
        chunk_size=(1, 500, 500)
    )
    print(f"   Created: {zarr_path}")
    
    # 4. Calculate forest metrics
    print("\n4. Calculating forest metrics...")
    results = api.calculate_metrics(
        zarr_path=zarr_path,
        calculations=["species_richness", "shannon_diversity", "total_biomass"],
        output_dir="results/wake_county"
    )
    print(f"   Completed {len(results)} calculations:")
    for r in results:
        print(f"     - {r.name}: {r.output_path.name}")
    
    # 5. Create visualizations
    print("\n5. Creating visualizations...")
    maps = api.create_maps(
        zarr_path=zarr_path,
        map_type="diversity",
        output_dir="maps/wake_county"
    )
    print(f"   Created {len(maps)} maps")
    
    return zarr_path


def example_location_configurations():
    """Working with different geographic locations."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Location Configurations")
    print("="*60)
    
    api = BigMapAPI()
    
    # State-level configuration
    print("\n1. State configuration (Montana)...")
    state_config = api.get_location_config(
        state="Montana",
        output_path="configs/montana.yaml"
    )
    print(f"   Location: {state_config.location_name}")
    print(f"   Bbox: {state_config.web_mercator_bbox}")
    
    # County-level configuration
    print("\n2. County configuration (Harris County, TX)...")
    county_config = api.get_location_config(
        state="Texas",
        county="Harris",
        output_path="configs/harris_county.yaml"
    )
    print(f"   Location: {county_config.location_name}")
    
    # Custom bounding box
    print("\n3. Custom bounding box configuration...")
    custom_config = api.get_location_config(
        bbox=(-104.5, 44.0, -104.0, 44.5),
        crs="EPSG:4326",
        output_path="configs/custom_area.yaml"
    )
    print(f"   Created custom configuration")


def example_advanced_analysis():
    """Advanced analysis with custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Advanced Analysis")
    print("="*60)
    
    # Create custom configuration
    from bigmap import BigMapSettings, CalculationConfig
    
    settings = BigMapSettings(
        output_dir=Path("output/advanced"),
        calculations=[
            CalculationConfig(name="species_richness", enabled=True),
            CalculationConfig(name="shannon_diversity", enabled=True),
            CalculationConfig(name="simpson_diversity", enabled=True),
            CalculationConfig(name="evenness", enabled=True),
            CalculationConfig(name="total_biomass", enabled=True),
            CalculationConfig(name="dominant_species", enabled=True),
        ]
    )
    
    # Initialize API with custom settings
    api = BigMapAPI(config=settings)
    
    # Assuming we have a zarr file
    zarr_path = Path("data/forest.zarr")
    
    if zarr_path.exists():
        print("\n1. Running comprehensive analysis...")
        results = api.calculate_metrics(zarr_path=zarr_path)
        print(f"   Completed {len(results)} calculations")
        
        # Validate the zarr store
        print("\n2. Validating Zarr store...")
        info = api.validate_zarr(zarr_path)
        print(f"   Shape: {info['shape']}")
        print(f"   Species: {info['num_species']}")
        print(f"   Chunks: {info.get('chunks', 'N/A')}")
    else:
        print(f"   Zarr file not found: {zarr_path}")
        print("   Run example_basic_workflow() first to create it")


def example_batch_processing():
    """Batch processing multiple locations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)
    
    api = BigMapAPI()
    
    # Define multiple locations to process
    locations = [
        {"state": "North Carolina", "counties": ["Wake", "Durham", "Orange"]},
        {"state": "Montana", "counties": ["Missoula", "Flathead"]},
    ]
    
    print("\nProcessing multiple locations:")
    
    for location in locations:
        state = location["state"]
        for county in location["counties"]:
            print(f"\n  Processing {county} County, {state}...")
            
            # Download species data
            output_dir = Path(f"data/{state.lower().replace(' ', '_')}/{county.lower()}")
            
            try:
                files = api.download_species(
                    state=state,
                    county=county,
                    species_codes=["0202", "0122"],  # Douglas-fir, Ponderosa Pine
                    output_dir=output_dir
                )
                print(f"    Downloaded {len(files)} species files")
                
                # Create Zarr store
                zarr_path = output_dir.parent / f"{county.lower()}.zarr"
                api.create_zarr(
                    input_dir=output_dir,
                    output_path=zarr_path
                )
                print(f"    Created Zarr store: {zarr_path.name}")
                
                # Calculate metrics
                results = api.calculate_metrics(
                    zarr_path=zarr_path,
                    calculations=["species_richness", "total_biomass"]
                )
                print(f"    Calculated {len(results)} metrics")
                
            except Exception as e:
                print(f"    Error: {e}")


def example_jupyter_notebook():
    """Example of using BigMap in Jupyter notebooks."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Jupyter Notebook Usage")
    print("="*60)
    
    print("\nBigMap is perfect for Jupyter notebooks:")
    print("-" * 40)
    print("""
# Cell 1: Import and initialize
from bigmap import BigMapAPI
import matplotlib.pyplot as plt

api = BigMapAPI()

# Cell 2: Explore available data
species = api.list_species()
print(f"Available species: {len(species)}")
species_df = pd.DataFrame([s.dict() for s in species])
species_df.head()

# Cell 3: Download and process
files = api.download_species(state="Montana", species_codes=["0202"])
zarr_path = api.create_zarr("downloads/", "data/montana.zarr")

# Cell 4: Analyze
results = api.calculate_metrics(
    zarr_path, 
    calculations=["species_richness", "shannon_diversity"]
)

# Cell 5: Visualize
maps = api.create_maps(zarr_path, map_type="diversity")
img = plt.imread(maps[0])
plt.imshow(img)
plt.axis('off')
plt.show()
""")
    
    print("\nAdvantages for Data Science:")
    print("  • Interactive exploration of data")
    print("  • Easy integration with pandas, numpy, matplotlib")
    print("  • Step-by-step analysis with intermediate results")
    print("  • Reproducible research workflows")
    print("  • Direct access to returned data structures")


def main():
    """Run all examples."""
    print("\n" + "🌲"*30)
    print("BigMap API Usage Examples")
    print("API-First Architecture Demonstration")
    print("🌲"*30)
    
    # Run examples (comment out any you don't want to run)
    
    # Basic workflow - downloads real data
    # zarr_path = example_basic_workflow()
    
    # Location configurations - just creates config files
    example_location_configurations()
    
    # Advanced analysis - requires existing data
    example_advanced_analysis()
    
    # Batch processing - downloads lots of data
    # example_batch_processing()
    
    # Jupyter notebook usage - just prints info
    example_jupyter_notebook()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\nThe BigMapAPI provides a clean, programmatic interface to all")
    print("BigMap functionality. Use it in scripts, Jupyter notebooks, or")
    print("integrate it into larger applications for forest analysis.")


if __name__ == "__main__":
    main()