#!/usr/bin/env python3
"""
Download checkpoints from Tinker remote storage to local directory.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import tinker


async def download_checkpoints(
    checkpoints_jsonl_path: str,
    output_dir: str,
):
    """Download all checkpoints listed in checkpoints.jsonl to local directory."""
    service_client = tinker.ServiceClient()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoints_jsonl_path, 'r') as f:
        checkpoints = [json.loads(line) for line in f]
    
    print(f"Found {len(checkpoints)} checkpoints to download")
    
    for checkpoint in checkpoints:
        name = checkpoint["name"]
        state_path = checkpoint["state_path"]
        sampler_path = checkpoint.get("sampler_path")
        
        print(f"\nDownloading checkpoint '{name}'...")
        print(f"  State path: {state_path}")
        
        checkpoint_dir = output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Download state (weights)
        try:
            state_data = await service_client.download_from_remote_path_async(state_path)
            state_file = checkpoint_dir / "state.pt"
            with open(state_file, 'wb') as f:
                f.write(state_data)
            size_mb = len(state_data) / (1024 * 1024)
            print(f"  ✓ Saved state: {state_file} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  ✗ Failed to download state: {e}")
        
        # Download sampler weights (optional)
        if sampler_path:
            try:
                sampler_data = await service_client.download_from_remote_path_async(sampler_path)
                sampler_file = checkpoint_dir / "sampler.pt"
                with open(sampler_file, 'wb') as f:
                    f.write(sampler_data)
                size_mb = len(sampler_data) / (1024 * 1024)
                print(f"  ✓ Saved sampler: {sampler_file} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  ✗ Failed to download sampler: {e}")
    
    print(f"\nCheckpoints saved to: {output_dir}")
    print_total_size(output_dir)


def print_total_size(directory: Path):
    """Print total size of directory."""
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)
    
    if size_gb >= 1:
        print(f"Total size: {size_gb:.2f} GB")
    else:
        print(f"Total size: {size_mb:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_checkpoints.py <checkpoints_jsonl_path> [output_dir]")
        print()
        print("Example:")
        print("  python download_checkpoints.py /tmp/tinker-examples/sl_basic/checkpoints.jsonl ./checkpoints")
        sys.exit(1)
    
    checkpoints_jsonl = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./checkpoints"
    
    asyncio.run(download_checkpoints(checkpoints_jsonl, output_dir))
