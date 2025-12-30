#!/usr/bin/env python3
"""
Upload State Zarr Stores to Cloud Storage

This script uploads processed state Zarr stores to cloud storage (R2 or B2).
It uses the manifest file to track what needs to be uploaded.

Supported backends:
- Cloudflare R2 (S3-compatible, zero egress fees)
- Backblaze B2 (S3-compatible, low cost)

Usage:
    # Upload all states to R2
    python scripts/upload_states_to_cloud.py \
        --manifest ./us_forest_data/manifest.json \
        --backend r2 \
        --bucket gridfia-states \
        --account-id YOUR_ACCOUNT_ID

    # Upload specific states to B2
    python scripts/upload_states_to_cloud.py \
        --manifest ./us_forest_data/manifest.json \
        --backend b2 \
        --bucket gridfia-states \
        --states NC VA GA

    # Dry run to see what would be uploaded
    python scripts/upload_states_to_cloud.py --manifest ./us_forest_data/manifest.json --dry-run

Prerequisites:
    - AWS CLI configured with appropriate credentials
    - For R2: aws configure --profile r2
    - For B2: aws configure --profile b2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def get_endpoint_url(backend: str, account_id: Optional[str] = None, region: str = "us-west-004") -> str:
    """Get the S3 endpoint URL for the backend."""
    if backend == 'r2':
        if not account_id:
            raise ValueError("account_id required for R2")
        return f"https://{account_id}.r2.cloudflarestorage.com"
    elif backend == 'b2':
        return f"https://s3.{region}.backblazeb2.com"
    else:
        raise ValueError(f"Unknown backend: {backend}")


def upload_zarr_to_cloud(
    local_path: Path,
    bucket: str,
    target_key: str,
    endpoint_url: str,
    profile: str = 'default',
    dry_run: bool = False
) -> bool:
    """
    Upload a Zarr store to cloud storage using AWS CLI.

    Returns True on success, False on failure.
    """
    s3_target = f"s3://{bucket}/{target_key}"

    console.print(f"  Uploading {local_path.name} -> {s3_target}")

    if dry_run:
        console.print(f"  [dim]DRY RUN: Would sync {local_path} to {s3_target}[/dim]")
        return True

    cmd = [
        "aws", "s3", "sync",
        str(local_path),
        s3_target,
        "--endpoint-url", endpoint_url,
        "--profile", profile,
        "--no-progress"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            console.print(f"  [red]Error: {result.stderr}[/red]")
            return False

        console.print(f"  [green]Success[/green]")
        return True

    except subprocess.TimeoutExpired:
        console.print(f"  [red]Upload timed out[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
        return False


def load_manifest(manifest_path: Path) -> Dict:
    """Load the manifest file."""
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found: {manifest_path}[/red]")
        sys.exit(1)

    with open(manifest_path) as f:
        return json.load(f)


def update_manifest_urls(
    manifest_path: Path,
    manifest: Dict,
    bucket: str,
    public_url: str,
    uploaded_states: List[str]
):
    """Update manifest with public URLs for uploaded states."""
    for state_abbr in uploaded_states:
        if state_abbr in manifest['states']:
            zarr_name = f"{state_abbr.lower()}_forest.zarr"
            manifest['states'][state_abbr]['url'] = f"{public_url}/states/{state_abbr.lower()}/{zarr_name}"

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[green]Updated manifest with URLs[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Upload state Zarr stores to cloud storage"
    )

    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('./us_forest_data/manifest.json'),
        help='Path to manifest.json'
    )
    parser.add_argument(
        '--backend',
        choices=['r2', 'b2'],
        default='b2',
        help='Cloud storage backend (default: b2)'
    )
    parser.add_argument(
        '--bucket',
        default='gridfia-data',
        help='Bucket name (default: gridfia-data)'
    )
    parser.add_argument(
        '--account-id',
        help='Account ID (required for R2)'
    )
    parser.add_argument(
        '--profile',
        default='b2',
        help='AWS CLI profile name (default: b2)'
    )
    parser.add_argument(
        '--public-url',
        help='Public URL for the bucket. '
             'B2: https://f004.backblazeb2.com/file/<bucket>, '
             'R2: https://pub-xxx.r2.dev'
    )
    parser.add_argument(
        '--region',
        default='us-west-004',
        help='B2 region (default: us-west-004)'
    )
    parser.add_argument(
        '--states',
        nargs='+',
        help='Specific states to upload (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )

    args = parser.parse_args()

    # Validation
    if args.backend == 'r2' and not args.account_id:
        console.print("[red]--account-id required for R2 backend[/red]")
        sys.exit(1)

    # Load manifest
    manifest = load_manifest(args.manifest)

    if not manifest.get('states'):
        console.print("[red]No states found in manifest[/red]")
        sys.exit(1)

    # Determine which states to upload
    available_states = list(manifest['states'].keys())
    if args.states:
        states_to_upload = [s.upper() for s in args.states if s.upper() in available_states]
        invalid = [s for s in args.states if s.upper() not in available_states]
        if invalid:
            console.print(f"[yellow]States not in manifest: {invalid}[/yellow]")
    else:
        states_to_upload = available_states

    if not states_to_upload:
        console.print("[red]No states to upload[/red]")
        sys.exit(1)

    # Get endpoint
    endpoint_url = get_endpoint_url(args.backend, args.account_id, args.region)

    # Generate default public URL if not provided
    if not args.public_url:
        if args.backend == 'b2':
            # B2 public URL format: https://f004.backblazeb2.com/file/<bucket>
            region_code = args.region.split('-')[-1]  # e.g., "004" from "us-west-004"
            args.public_url = f"https://f{region_code}.backblazeb2.com/file/{args.bucket}"
        else:
            console.print("[yellow]Warning: No --public-url specified for R2[/yellow]")

    console.print(f"[bold]Uploading {len(states_to_upload)} states to {args.backend.upper()}[/bold]")
    console.print(f"  Bucket: {args.bucket}")
    console.print(f"  Endpoint: {endpoint_url}")
    console.print(f"  Public URL: {args.public_url}")
    console.print(f"  Profile: {args.profile}")
    if args.dry_run:
        console.print(f"  [yellow]DRY RUN MODE[/yellow]")
    console.print()

    # Upload each state
    uploaded = []
    failed = []

    for state_abbr in states_to_upload:
        state_info = manifest['states'][state_abbr]
        local_path = Path(state_info.get('local_path', ''))

        if not local_path.exists():
            console.print(f"[red]{state_abbr}: Local path not found: {local_path}[/red]")
            failed.append(state_abbr)
            continue

        target_key = f"states/{state_abbr.lower()}/{state_abbr.lower()}_forest.zarr"

        success = upload_zarr_to_cloud(
            local_path=local_path,
            bucket=args.bucket,
            target_key=target_key,
            endpoint_url=endpoint_url,
            profile=args.profile,
            dry_run=args.dry_run
        )

        if success:
            uploaded.append(state_abbr)
        else:
            failed.append(state_abbr)

    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold]Upload Summary[/bold]")
    console.print(f"  Uploaded: {len(uploaded)} states")
    console.print(f"  Failed: {len(failed)} states")

    if uploaded:
        console.print(f"  [green]Success: {', '.join(sorted(uploaded))}[/green]")
    if failed:
        console.print(f"  [red]Failed: {', '.join(sorted(failed))}[/red]")

    # Update manifest with URLs
    if uploaded and args.public_url and not args.dry_run:
        update_manifest_urls(
            args.manifest,
            manifest,
            args.bucket,
            args.public_url,
            uploaded
        )


if __name__ == '__main__':
    main()
