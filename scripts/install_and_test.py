#!/usr/bin/env python3
"""
Installation and testing script for BigMap REST API functionality.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Run installation and tests."""
    print("🚀 BigMap REST API Installation and Testing")
    print("=" * 50)
    
    # Step 1: Reinstall package
    if not run_command(
        "source .venv/bin/activate && pip install -e .",
        "Reinstalling BigMap package with new dependencies"
    ):
        print("❌ Failed to install package")
        return False
    
    # Step 2: Test basic CLI
    if not run_command(
        "source .venv/bin/activate && bigmap --help",
        "Testing main CLI command"
    ):
        print("❌ Main CLI command failed")
        return False
    
    # Step 3: Test REST API connection
    if not run_command(
        "source .venv/bin/activate && python test_rest_api.py",
        "Testing REST API client connection"
    ):
        print("❌ REST API test failed")
        return False
    
    # Step 4: Test new CLI commands
    commands_to_test = [
        ("bigmap list-api-species", "List available species via REST API"),
        ("bigmap identify-point -s 0131 -x -8750000 -y 4285000", "Identify biomass at a point"),
        ("bigmap species-stats -s 0131", "Get species statistics"),
    ]
    
    for cmd, desc in commands_to_test:
        if not run_command(f"source .venv/bin/activate && {cmd}", desc):
            print(f"⚠️  Command '{cmd}' failed, but continuing...")
    
    # Step 5: Test small download
    print("\n🔄 Testing small species download...")
    if run_command(
        "source .venv/bin/activate && bigmap download-species-api -s 0131 --output-dir test_download",
        "Download Loblolly Pine data via REST API"
    ):
        print("✅ Download test successful!")
        
        # Check if file was created
        test_files = list(Path("test_download").glob("*.tif"))
        if test_files:
            print(f"✅ Downloaded file: {test_files[0]}")
        else:
            print("⚠️  No files found in test_download directory")
    
    print("\n" + "=" * 50)
    print("🎉 Installation and testing complete!")
    print("\nNext steps:")
    print("1. Use 'bigmap list-api-species' to see all 325+ available species")
    print("2. Use 'bigmap download-species-api -s <codes>' to download specific species")
    print("3. Use 'bigmap identify-point' to get biomass values at specific locations")
    print("4. Use 'bigmap species-stats' to get statistics for species")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 