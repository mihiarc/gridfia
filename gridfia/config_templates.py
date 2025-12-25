"""
Configuration template generator for BigMap.

This module provides functions to generate configuration file templates
for different types of analyses.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from .config import GridFIASettings, CalculationConfig


def create_config_template(
    config_type: str,
    output_path: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a configuration template for a specific analysis type.
    
    Parameters
    ----------
    config_type : str
        Type of configuration: 'analysis', 'species', 'data'
    output_path : Path, optional
        If provided, save the configuration to this path
    **kwargs : dict
        Additional parameters for the configuration
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    templates = {
        'analysis': _create_analysis_template,
        'species': _create_species_template,
        'data': _create_data_template,
    }
    
    if config_type not in templates:
        raise ValueError(f"Unknown config type: {config_type}. "
                        f"Choose from: {list(templates.keys())}")
    
    config = templates[config_type](**kwargs)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def _create_analysis_template(
    name: str = "custom_analysis",
    description: str = "Custom forest analysis configuration",
    calculations: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a general analysis configuration template."""
    if calculations is None:
        calculations = ["species_richness", "total_biomass"]
    
    config = {
        'name': name,
        'description': description,
        'output_dir': f"output/{name}",
        'calculations': []
    }
    
    # Add calculation configurations
    calc_templates = {
        'species_richness': {
            'name': 'species_richness',
            'enabled': True,
            'parameters': {
                'biomass_threshold': 0.0,
                'exclude_total_layer': True
            }
        },
        'total_biomass': {
            'name': 'total_biomass',
            'enabled': True,
            'parameters': {
                'exclude_total_layer': True
            }
        },
        'shannon_diversity': {
            'name': 'shannon_diversity',
            'enabled': True,
            'parameters': {
                'exclude_total_layer': True,
                'base': 'e'
            }
        },
        'simpson_diversity': {
            'name': 'simpson_diversity',
            'enabled': True,
            'parameters': {
                'exclude_total_layer': True,
                'inverse': True
            }
        },
        'evenness': {
            'name': 'evenness',
            'enabled': True,
            'parameters': {
                'exclude_total_layer': True
            }
        },
        'dominant_species': {
            'name': 'dominant_species',
            'enabled': True,
            'parameters': {
                'exclude_total_layer': True,
                'min_biomass': 0.0
            }
        }
    }
    
    for calc in calculations:
        if calc in calc_templates:
            config['calculations'].append(calc_templates[calc])
    
    # Add any additional parameters
    config.update(kwargs)
    
    return config


def _create_species_template(
    species_codes: Optional[List[int]] = None,
    species_names: Optional[List[str]] = None,
    group_name: str = "species_group",
    **kwargs
) -> Dict[str, Any]:
    """Create a species-specific configuration template."""
    if species_codes is None:
        species_codes = [131]  # Default to Loblolly Pine
    if species_names is None:
        species_names = ["Loblolly Pine"]
    
    config = {
        'name': f"{group_name}_analysis",
        'description': f"Analysis of {group_name}",
        'output_dir': f"output/{group_name}",
        'species': {
            'codes': species_codes,
            'names': species_names,
            'group_name': group_name
        },
        'calculations': [
            {
                'name': 'species_proportion',
                'enabled': True,
                'parameters': {
                    'species_indices': [i+1 for i in range(len(species_codes))],
                    'species_names': species_names
                }
            },
            {
                'name': 'species_group_proportion',
                'enabled': True,
                'parameters': {
                    'species_indices': [i+1 for i in range(len(species_codes))],
                    'group_name': group_name
                }
            }
        ]
    }
    
    config.update(kwargs)
    return config




def _create_data_template(
    data_type: str = "raster",
    input_pattern: str = "*.tif",
    **kwargs
) -> Dict[str, Any]:
    """Create a data processing configuration template."""
    config = {
        'name': f"{data_type}_processing",
        'description': f"Process {data_type} data",
        'output_dir': f"output/{data_type}_processed",
        'data_processing': {
            'type': data_type,
            'input_pattern': input_pattern,
            'chunk_size': [1000, 1000],
            'compression': 'lz4',
            'overwrite': False
        }
    }
    
    if data_type == "raster":
        config['data_processing'].update({
            'crs': 'EPSG:102039',
            'pixel_size_meters': 30.0,
            'nodata_value': -9999
        })
    elif data_type == "vector":
        config['data_processing'].update({
            'geometry_column': 'geometry',
            'id_column': 'parcel_id'
        })
    
    config.update(kwargs)
    return config


def list_available_templates() -> List[str]:
    """List all available configuration templates."""
    return ['analysis', 'species', 'data']


def generate_example_configs(output_dir: Path = Path("cfg/examples")):
    """Generate example configuration files for all template types."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    examples = {
        'basic_analysis_config.yaml': {
            'type': 'analysis',
            'params': {
                'name': 'basic_forest_analysis',
                'description': 'Basic forest metrics analysis',
                'calculations': ['species_richness', 'total_biomass']
            }
        },
        'pine_species_config.yaml': {
            'type': 'species',
            'params': {
                'species_codes': [131, 318, 111, 110],
                'species_names': ['Loblolly Pine', 'Longleaf Pine', 
                                'Shortleaf Pine', 'Slash Pine'],
                'group_name': 'southern_yellow_pine'
            }
        },
        'raster_processing_config.yaml': {
            'type': 'data',
            'params': {
                'data_type': 'raster',
                'input_pattern': 'nc_*.tif'
            }
        }
    }
    
    for filename, spec in examples.items():
        create_config_template(
            spec['type'],
            output_path=output_dir / filename,
            **spec['params']
        )
    
    print(f"Generated {len(examples)} example configurations in {output_dir}")


if __name__ == "__main__":
    # Generate example configurations
    generate_example_configs()