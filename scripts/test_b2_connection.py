#!/usr/bin/env python
"""
Test Backblaze B2 connection and credentials.

Usage:
    # Set environment variables first:
    export B2_ENDPOINT_URL="https://s3.us-west-004.backblazeb2.com"
    export B2_ACCESS_KEY_ID="your-application-key-id"
    export B2_SECRET_ACCESS_KEY="your-application-key"
    export B2_BUCKET="your-bucket-name"
    export B2_PUBLIC_URL="https://your-custom-domain.com"  # optional

    # Then run:
    python scripts/test_b2_connection.py
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from gridfia.conus import CloudStorage

console = Console()


def main():
    console.print(Panel("[bold]Backblaze B2 Connection Test[/bold]"))

    # Check environment variables
    console.print("\n[cyan]Checking environment variables...[/cyan]")

    env_vars = {
        "B2_ENDPOINT_URL": os.environ.get("B2_ENDPOINT_URL"),
        "B2_ACCESS_KEY_ID": os.environ.get("B2_ACCESS_KEY_ID"),
        "B2_SECRET_ACCESS_KEY": os.environ.get("B2_SECRET_ACCESS_KEY"),
        "B2_BUCKET": os.environ.get("B2_BUCKET", "gridfia-conus (default)"),
        "B2_PUBLIC_URL": os.environ.get("B2_PUBLIC_URL", "(not set - will use B2 URL)"),
    }

    required = ["B2_ENDPOINT_URL", "B2_ACCESS_KEY_ID", "B2_SECRET_ACCESS_KEY"]
    missing = []

    for name, value in env_vars.items():
        if value and "default" not in str(value) and "not set" not in str(value):
            if "SECRET" in name:
                masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            else:
                masked = value
            console.print(f"  {name}: [green]{masked}[/green]")
        elif "default" in str(value) or "not set" in str(value):
            console.print(f"  {name}: [yellow]{value}[/yellow]")
        else:
            console.print(f"  {name}: [red]NOT SET[/red]")
            if name in required:
                missing.append(name)

    if missing:
        console.print(f"\n[red]Missing required variables: {', '.join(missing)}[/red]")
        console.print("\nSet them with:")
        console.print("  export B2_ENDPOINT_URL='https://s3.us-west-004.backblazeb2.com'")
        console.print("  export B2_ACCESS_KEY_ID='your-application-key-id'")
        console.print("  export B2_SECRET_ACCESS_KEY='your-application-key'")
        console.print("  export B2_BUCKET='your-bucket-name'")
        console.print("  export B2_PUBLIC_URL='https://your-custom-domain.com'  # optional")
        sys.exit(1)

    # Test connection
    console.print("\n[cyan]Testing B2 connection...[/cyan]")

    try:
        storage = CloudStorage()

        if storage.client is None:
            console.print("[red]Failed to initialize B2 client[/red]")
            sys.exit(1)

        console.print(f"  Bucket: [green]{storage.bucket}[/green]")
        console.print(f"  Prefix: [green]{storage.prefix}[/green]")

        # Try to list objects (limited to 5)
        console.print("\n[cyan]Testing bucket access...[/cyan]")
        response = storage.client.list_objects_v2(
            Bucket=storage.bucket,
            Prefix=storage.prefix,
            MaxKeys=5,
        )

        count = response.get("KeyCount", 0)
        console.print(f"  Objects in prefix: [green]{count}[/green]")

        if count > 0:
            console.print("  Sample objects:")
            for obj in response.get("Contents", [])[:3]:
                console.print(f"    - {obj['Key']}")

        # Test write access with a small test file
        console.print("\n[cyan]Testing write access...[/cyan]")
        test_key = f"{storage.prefix}/_test_connection.txt"

        storage.client.put_object(
            Bucket=storage.bucket,
            Key=test_key,
            Body=b"test",
        )
        console.print(f"  Write test: [green]SUCCESS[/green]")

        # Clean up test file
        storage.client.delete_object(Bucket=storage.bucket, Key=test_key)
        console.print(f"  Cleanup: [green]SUCCESS[/green]")

        console.print("\n[bold green]B2 connection test passed![/bold green]")

        # Show public URL
        sample_url = storage.get_tile_url("conus_000_000")
        console.print(f"\nSample tile URL: {sample_url}")

    except Exception as e:
        console.print(f"\n[red]Connection failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
