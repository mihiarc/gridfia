"""
Cloud Storage for CONUS Tile System.

Handles upload/download of tiles to/from Cloudflare R2.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging
import hashlib
import json
import os

import boto3
from botocore.config import Config
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class CloudStorage:
    """
    Manages tile storage on Backblaze B2 (S3-compatible API).

    Uses S3-compatible API for uploads and downloads.
    """

    # Default B2 configuration
    DEFAULT_BUCKET = "gridfia-conus"
    DEFAULT_PREFIX = "tiles"

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        public_url_base: Optional[str] = None,
    ):
        """
        Initialize cloud storage client for Backblaze B2.

        Credentials can be provided directly or via environment variables:
        - B2_ENDPOINT_URL (e.g., https://s3.us-east-005.backblazeb2.com)
        - B2_ACCESS_KEY_ID (B2 application key ID)
        - B2_SECRET_ACCESS_KEY (B2 application key)
        - B2_BUCKET (bucket name)
        - B2_PUBLIC_URL (custom domain or B2 friendly URL)

        Args:
            bucket: B2 bucket name
            prefix: Key prefix for tiles
            endpoint_url: B2 S3-compatible endpoint
            access_key_id: B2 application key ID
            secret_access_key: B2 application key
            public_url_base: Public URL for accessing files
        """
        self.bucket = bucket or os.environ.get("B2_BUCKET", self.DEFAULT_BUCKET)
        self.prefix = prefix or os.environ.get("B2_PREFIX", self.DEFAULT_PREFIX)
        self.public_url_base = public_url_base or os.environ.get("B2_PUBLIC_URL", "")

        # Store endpoint for URL generation
        self._endpoint_url = endpoint_url or os.environ.get("B2_ENDPOINT_URL", "")

        endpoint_url = endpoint_url or os.environ.get("B2_ENDPOINT_URL")
        access_key_id = access_key_id or os.environ.get("B2_ACCESS_KEY_ID")
        secret_access_key = secret_access_key or os.environ.get("B2_SECRET_ACCESS_KEY")

        if not all([endpoint_url, access_key_id, secret_access_key]):
            logger.warning("B2 credentials not configured - uploads will fail")
            self.client = None
        else:
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )

    def _get_tile_key(self, tile_id: str, filename: str) -> str:
        """Get S3 key for a tile file."""
        return f"{self.prefix}/{tile_id}/{filename}"

    def upload_tile(
        self,
        tile_id: str,
        zarr_path: Path,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload a tile Zarr store to R2.

        Args:
            tile_id: Tile identifier
            zarr_path: Local path to Zarr store
            show_progress: Show progress bar

        Returns:
            Upload metadata including checksums
        """
        if self.client is None:
            raise RuntimeError("R2 client not configured")

        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

        # Get all files to upload
        files = list(zarr_path.rglob("*"))
        files = [f for f in files if f.is_file()]

        uploaded_files = []
        total_size = 0
        checksums = {}

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[blue]Uploading {tile_id}",
                    total=len(files),
                )

                for file_path in files:
                    relative_path = file_path.relative_to(zarr_path)
                    key = self._get_tile_key(tile_id, f"biomass.zarr/{relative_path}")

                    # Calculate checksum for verification
                    md5_hash = self._calculate_md5(file_path)
                    checksums[str(relative_path)] = md5_hash

                    # Upload (S3 Transfer Manager handles integrity automatically)
                    with open(file_path, "rb") as f:
                        self.client.upload_fileobj(f, self.bucket, key)

                    uploaded_files.append(key)
                    total_size += file_path.stat().st_size
                    progress.update(task, advance=1)
        else:
            for file_path in files:
                relative_path = file_path.relative_to(zarr_path)
                key = self._get_tile_key(tile_id, f"biomass.zarr/{relative_path}")

                md5_hash = self._calculate_md5(file_path)
                checksums[str(relative_path)] = md5_hash

                with open(file_path, "rb") as f:
                    self.client.upload_fileobj(f, self.bucket, key)

                uploaded_files.append(key)
                total_size += file_path.stat().st_size

        return {
            "tile_id": tile_id,
            "files_uploaded": len(uploaded_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "checksums": checksums,
        }

    def verify_tile(self, tile_id: str) -> bool:
        """
        Verify a tile exists and is accessible on R2.

        Args:
            tile_id: Tile identifier

        Returns:
            True if tile is accessible
        """
        if self.client is None:
            return False

        try:
            # Check for zarr.json (root metadata file)
            key = self._get_tile_key(tile_id, "biomass.zarr/zarr.json")
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def get_tile_url(self, tile_id: str) -> str:
        """
        Get public URL for a tile.

        Args:
            tile_id: Tile identifier

        Returns:
            Public URL for the tile Zarr store
        """
        if self.public_url_base:
            return f"{self.public_url_base}/{self.prefix}/{tile_id}/biomass.zarr"
        elif self._endpoint_url:
            # Extract region from endpoint URL (e.g., s3.us-east-005.backblazeb2.com)
            region = self._endpoint_url.replace("https://", "").replace("http://", "")
            return f"https://{self.bucket}.{region}/{self.prefix}/{tile_id}/biomass.zarr"
        else:
            # Fallback to generic B2 URL format
            return f"https://{self.bucket}.s3.us-east-005.backblazeb2.com/{self.prefix}/{tile_id}/biomass.zarr"

    def upload_index(self, index_data: Dict[str, Any], filename: str) -> str:
        """
        Upload an index file to R2.

        Args:
            index_data: Index data dictionary
            filename: Index filename (e.g., conus_tile_index.json)

        Returns:
            S3 key of uploaded file
        """
        if self.client is None:
            raise RuntimeError("R2 client not configured")

        key = f"conus/index/{filename}"
        json_data = json.dumps(index_data, indent=2).encode("utf-8")

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json_data,
            ContentType="application/json",
        )

        logger.info(f"Uploaded index: {key}")
        return key

    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _md5_to_base64(self, md5_hex: str) -> str:
        """Convert MD5 hex to base64 for S3 ContentMD5."""
        import base64

        return base64.b64encode(bytes.fromhex(md5_hex)).decode("utf-8")

    def list_uploaded_tiles(self) -> list[str]:
        """
        List all uploaded tile IDs.

        Returns:
            List of tile IDs that have been uploaded
        """
        if self.client is None:
            return []

        tiles = set()
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract tile ID from key like conus/tiles/conus_000_000/biomass.zarr/...
                parts = key.split("/")
                if len(parts) >= 3 and parts[2].startswith("conus_"):
                    tiles.add(parts[2])

        return sorted(tiles)
