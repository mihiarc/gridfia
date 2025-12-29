#!/usr/bin/env python3
"""
Upload sample Zarr datasets to Cloudflare R2.

This script uploads GridFIA sample datasets to R2 for public access.
Requires AWS CLI configured with R2 credentials.

Setup:
1. Create R2 bucket in Cloudflare dashboard
2. Get R2 API credentials
3. Configure AWS CLI:
   aws configure --profile r2
   # Enter R2 Access Key ID and Secret Access Key
   # Region: auto
   # Output: json

Usage:
   python scripts/upload_to_r2.py --bucket pub-gridfia --zarr-path examples/durham_data/durham_forest.zarr
"""

import argparse
import subprocess
import sys
from pathlib import Path


def upload_zarr_to_r2(
    zarr_path: Path,
    bucket: str,
    target_name: str,
    account_id: str,
    profile: str = "r2"
) -> str:
    """
    Upload a Zarr store to Cloudflare R2.

    Args:
        zarr_path: Local path to Zarr store
        bucket: R2 bucket name
        target_name: Name for the dataset in R2 (e.g., "durham_nc")
        account_id: Cloudflare account ID
        profile: AWS CLI profile name for R2 credentials

    Returns:
        Public URL for the uploaded dataset
    """
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    # R2 endpoint URL
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    # Target path in R2
    s3_target = f"s3://{bucket}/samples/{target_name}.zarr"

    print(f"Uploading {zarr_path} to {s3_target}")
    print(f"This may take a few minutes for large datasets...")

    # Use AWS CLI to sync the Zarr directory to R2
    cmd = [
        "aws", "s3", "sync",
        str(zarr_path),
        s3_target,
        "--endpoint-url", endpoint_url,
        "--profile", profile,
        "--no-progress"  # Cleaner output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error uploading: {result.stderr}")
        sys.exit(1)

    print(f"Upload complete!")

    # Public URL format for R2 with public bucket access
    public_url = f"https://pub-{bucket}.r2.dev/samples/{target_name}.zarr"

    return public_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload GridFIA sample datasets to Cloudflare R2"
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to local Zarr store"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="gridfia",
        help="R2 bucket name (default: gridfia)"
    )
    parser.add_argument(
        "--target-name",
        type=str,
        required=True,
        help="Name for the dataset (e.g., 'durham_nc')"
    )
    parser.add_argument(
        "--account-id",
        type=str,
        required=True,
        help="Cloudflare account ID"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="r2",
        help="AWS CLI profile for R2 credentials (default: r2)"
    )

    args = parser.parse_args()

    url = upload_zarr_to_r2(
        zarr_path=args.zarr_path,
        bucket=args.bucket,
        target_name=args.target_name,
        account_id=args.account_id,
        profile=args.profile
    )

    print(f"\nPublic URL: {url}")
    print("\nUpdate SAMPLE_DATASETS in gridfia/api.py with this URL")


if __name__ == "__main__":
    main()
