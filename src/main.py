#!/usr/bin/env python3
"""
Main entry point for the heirs property analysis pipeline.
"""

from pathlib import Path
from data_processing.processor import DataProcessor
from config import Config

def main():
    """Run the complete pipeline."""
    # Initialize configuration
    config = Config()
    
    # Initialize and run processor
    processor = DataProcessor(config)
    results = processor.run()
    
    # Print results
    if results["status"] == "success":
        print("\nProcessing completed successfully!")
        print(f"Properties processed: {results['properties_processed']}")
        print(f"Output saved to: {results['output_file']}")
    else:
        print(f"\nProcessing failed: {results['error']}")

if __name__ == "__main__":
    main() 