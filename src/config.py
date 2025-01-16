from pathlib import Path
from typing import Dict
import yaml

class Config:
    """Simple configuration management for the pipeline."""
    
    def __init__(self, config_path: str | Path = None):
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self.settings = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Default configuration
            return {
                "input_dir": "data/raw",
                "output_dir": "data/processed",
                "years": list(range(2015, 2024)),
                "required_fields": ["parcel_id", "county", "area"]
            }
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    @property
    def input_dir(self) -> Path:
        return Path(self.settings["input_dir"])
    
    @property
    def output_dir(self) -> Path:
        return Path(self.settings["output_dir"])
    
    @property
    def years(self) -> list:
        return self.settings["years"]
    
    @property
    def required_fields(self) -> list:
        return self.settings["required_fields"] 