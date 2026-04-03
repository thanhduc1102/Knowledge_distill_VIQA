"""
Upload pipeline code and dependencies to Kaggle as datasets for offline execution.
Uses kagglehub API for reliable uploads.

Usage:
    python scripts/kaggle_upload.py --upload-code
    python scripts/kaggle_upload.py --upload-wheels
    python scripts/kaggle_upload.py --all
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path

# Load credentials
def load_credentials():
    """Load Kaggle credentials from .env or environment."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line or line.startswith('os.'):
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'HF_TOKEN'):
                    os.environ[key] = val

    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    if not username or not key:
        print("ERROR: Set KAGGLE_USERNAME and KAGGLE_KEY in .env or environment")
        sys.exit(1)

    # Set up kaggle credentials file
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json, 0o600)

    return username


def prepare_code_package(output_dir: Path):
    """Package pipeline code for Kaggle upload."""
    project_root = Path(__file__).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directories to include
    dirs_to_copy = ['pipeline', 'src', 'configs']
    files_to_copy = ['requirements.txt']

    for d in dirs_to_copy:
        src = project_root / d
        dst = output_dir / d
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
            print(f"  Copied {d}/")

    for f in files_to_copy:
        src = project_root / f
        dst = output_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {f}")

    print(f"Code package prepared: {output_dir}")


def download_wheels(output_dir: Path):
    """Download Python wheels for offline installation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    packages = [
        "transformers>=5.0",
        "peft>=0.18",
        "accelerate>=1.0",
        "datasets>=4.0",
        "bitsandbytes>=0.49",
        "trl>=1.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "sympy",
        "pyyaml",
        "pyarrow",
        "tqdm",
        "pandas",
        "huggingface_hub>=1.0",
        "tokenizers",
    ]

    print(f"Downloading wheels to {output_dir}...")
    for pkg in packages:
        print(f"  Downloading {pkg}...")
        result = subprocess.run(
            ["pip", "download", pkg, "-d", str(output_dir),
             "--only-binary=:all:", "--python-version=3.12",
             "--platform=manylinux2014_x86_64", "--no-deps"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Try without platform restriction
            subprocess.run(
                ["pip", "download", pkg, "-d", str(output_dir), "--no-deps"],
                capture_output=True, text=True
            )

    wheel_count = len(list(output_dir.glob("*.whl"))) + len(list(output_dir.glob("*.tar.gz")))
    print(f"Downloaded {wheel_count} packages to {output_dir}")


def upload_dataset(dataset_slug: str, source_dir: Path, title: str):
    """Upload a directory as a Kaggle dataset using kaggle API."""
    # Create dataset-metadata.json
    metadata = {
        "title": title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    metadata_path = source_dir / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Uploading {dataset_slug}...")
    # Try create first, then update
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(source_dir), "--dir-mode", "zip"],
        capture_output=True, text=True
    )
    if "already exists" in result.stderr or result.returncode != 0:
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(source_dir),
             "-m", "Updated", "--dir-mode", "zip"],
            capture_output=True, text=True
        )

    if result.returncode == 0:
        print(f"  Upload successful: {dataset_slug}")
    else:
        print(f"  Upload output: {result.stdout}")
        if result.stderr:
            print(f"  Upload errors: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Upload pipeline to Kaggle")
    parser.add_argument("--upload-code", action="store_true", help="Upload pipeline code")
    parser.add_argument("--upload-wheels", action="store_true", help="Upload Python wheels")
    parser.add_argument("--all", action="store_true", help="Upload everything")
    parser.add_argument("--output-dir", default="/tmp/kaggle_upload", help="Temp directory")
    args = parser.parse_args()

    if not (args.upload_code or args.upload_wheels or args.all):
        parser.print_help()
        return

    username = load_credentials()
    output_base = Path(args.output_dir)

    if args.upload_code or args.all:
        print("\n=== Preparing Pipeline Code ===")
        code_dir = output_base / "vlsp2025-kd-pipeline"
        prepare_code_package(code_dir)
        upload_dataset(f"{username}/vlsp2025-kd-pipeline", code_dir, "VLSP2025 KD Pipeline Code")

    if args.upload_wheels or args.all:
        print("\n=== Preparing Python Wheels ===")
        wheels_dir = output_base / "vlsp2025-kd-wheels"
        download_wheels(wheels_dir)
        upload_dataset(f"{username}/vlsp2025-kd-wheels", wheels_dir, "VLSP2025 KD Python Wheels")

    print("\n=== Upload Complete ===")
    print(f"Datasets available at: https://www.kaggle.com/{username}/datasets")
    print("\nIn Kaggle notebook, add these as input datasets:")
    print(f"  - {username}/vlsp2025-kd-pipeline")
    print(f"  - {username}/vlsp2025-kd-wheels")
    print(f"  - {username}/finqa-en")
    print(f"  - {username}/vinumericalqa-private")
    print(f"  - {username}/qwen_35_27b (or qwen_35_4b)")


if __name__ == "__main__":
    main()
