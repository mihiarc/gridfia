import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec

def load_and_visualize_ndvi(file_path):
    """
    Load and return NDVI data within valid range (-1 to 1)
    """
    with rasterio.open(file_path) as src:
        # Read data efficiently using masked arrays
        ndvi = src.read(1, masked=True)
        # Clip values to valid NDVI range
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi

def create_comparison_plot(ndvi_files, output_path='ndvi_comparison.png'):
    """
    Create a grid plot of NDVI data from multiple years
    """
    # Set up the figure with GridSpec for better control
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Common colormap for all plots
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    
    for idx, file_path in enumerate(sorted(ndvi_files)):
        # Get year from filename (assuming format contains year)
        year = Path(file_path).stem.split('_')[1]  # Adjust split based on filename pattern
        
        # Create subplot
        ax = fig.add_subplot(gs[idx//3, idx%3])
        
        # Load and plot NDVI
        ndvi = load_and_visualize_ndvi(file_path)
        im = ax.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f'NDVI {year}')
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='NDVI')
    
    # Adjust layout and save
    plt.suptitle('Vance County NDVI Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Define path to NDVI files
    ndvi_dir = Path('ndvi_layers/ndvi_Heirs-Vance_County')
    ndvi_files = list(ndvi_dir.glob('*.tif'))  # Adjust file pattern as needed
    
    # Create visualization
    create_comparison_plot(ndvi_files) 