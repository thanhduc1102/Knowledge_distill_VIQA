"""
Upload pipeline code and dependencies to Kaggle as datasets for offline execution.
Uses kagglehub API for reliable uploads.

Usage:
    python scripts/kaggle_upload.py --upload-code
    python scripts/kaggle_upload.py --upload-wheels
    python scripts/kaggle_upload.py --upload-data
    python scripts/kaggle_upload.py --upload-models
    python scripts/kaggle_upload.py --check
    python scripts/kaggle_upload.py --all
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path


def _strip_inline_comment(val: str) -> str:
    """Strip inline comments from .env values.
    Handles: 'value  # comment' → 'value'
    Preserves values that start with # (already filtered upstream).
    """
    # Find the first # that is preceded by whitespace (inline comment)
    idx = val.find('#')
    while idx > 0:
        # Only treat as comment if preceded by whitespace
        if val[idx - 1] in (' ', '\t'):
            return val[:idx].rstrip()
        idx = val.find('#', idx + 1)
    return val


def _parse_env_line(line: str):
    """Parse a single .env line, returning (key, value) or None.
    Supports both 'KEY = value' and 'os.environ["KEY"] = value' formats.
    """
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        return None

    # Handle os.environ['KEY'] = value format
    if line.startswith('os.environ'):
        # Extract key from os.environ['KEY'] or os.environ["KEY"]
        try:
            bracket_start = line.index('[')
            bracket_end = line.index(']')
            key = line[bracket_start + 1:bracket_end].strip("'\"")
            val = line.split('=', 1)[1].strip().strip('"').strip("'")
            val = _strip_inline_comment(val)
            return key, val
        except (ValueError, IndexError):
            return None

    key, val = line.split('=', 1)
    key = key.strip()
    val = val.strip().strip('"').strip("'")
    val = _strip_inline_comment(val)
    return key, val


def load_credentials():
    """Load Kaggle credentials from .env or environment."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                result = _parse_env_line(line)
                if result is None:
                    continue
                key, val = result
                if key in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'HF_TOKEN'):
                    os.environ[key] = val

    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    if not username or not key:
        print("ERROR: Set KAGGLE_USERNAME and KAGGLE_KEY in .env or environment")
        sys.exit(1)

    # Validate credentials look reasonable
    if ' ' in username or len(username) > 50:
        print(f"WARNING: Username looks invalid: '{username}'")
        print("  Check your .env file - inline comments may not be stripped properly")
        sys.exit(1)

    # Set up kaggle credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json, 0o600)

    print(f"Kaggle credentials loaded for user: {username}")
    return username


def prepare_code_package(output_dir: Path):
    """Package pipeline code for Kaggle upload."""
    project_root = Path(__file__).parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    dirs_to_copy = ['pipeline', 'src', 'configs']
    files_to_copy = ['requirements.txt']

    for d in dirs_to_copy:
        src = project_root / d
        dst = output_dir / d
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  Copied {d}/")

    for f in files_to_copy:
        src = project_root / f
        dst = output_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {f}")

    # Count files
    total_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
    print(f"  Total files: {total_files}")
    return output_dir


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
        print(f"  {pkg}...", end=" ", flush=True)
        result = subprocess.run(
            ["pip", "download", pkg, "-d", str(output_dir), "--no-deps"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"WARN: {result.stderr[:100]}")

    wheel_count = len(list(output_dir.glob("*.whl"))) + len(list(output_dir.glob("*.tar.gz")))
    print(f"\nTotal packages: {wheel_count}")


def upload_dataset_kagglehub(handle: str, local_dir: str, version_notes: str = "Updated"):
    """Upload directory as Kaggle dataset using kagglehub."""
    import kagglehub

    print(f"\nUploading dataset: {handle}")
    print(f"  Source: {local_dir}")
    file_count = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
    print(f"  Files: {file_count}")

    try:
        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=local_dir,
            version_notes=version_notes,
        )
        print(f"  SUCCESS: {handle} uploaded!")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR: {error_msg}")
        if "403" in error_msg:
            print("  → 403 Forbidden: Check your KAGGLE_KEY in .env")
            print("    Make sure you use the API key from kaggle.json (not a KGAT_ token)")
            print("    Download fresh kaggle.json from: https://www.kaggle.com/settings → API → Create New Token")
        return False


def upload_model_kagglehub(handle: str, local_dir: str, license_name: str = "apache-2.0", version_notes: str = "Updated"):
    """Upload directory as Kaggle model using kagglehub."""
    import kagglehub

    print(f"\nUploading model: {handle}")
    print(f"  Source: {local_dir}")

    try:
        kagglehub.model_upload(
            handle=handle,
            local_model_dir=local_dir,
            license_name=license_name,
            version_notes=version_notes,
        )
        print(f"  SUCCESS: {handle} uploaded!")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"  ERROR: {error_msg}")
        if "403" in error_msg:
            print("  → 403 Forbidden: Check your KAGGLE_KEY (use classic API key, not KGAT_ token)")
        return False


def check_uploads(username: str):
    """Verify uploads exist on Kaggle."""
    import kagglehub

    datasets_to_check = [
        f"{username}/vlsp2025-kd-pipeline",
        f"{username}/vlsp2025-kd-wheels",
        f"{username}/finqa-en",
        f"{username}/vinumericalqa-private",
    ]

    # kagglehub model handles require 4 parts: owner/model/framework/variation
    models_to_check = [
        f"{username}/qwen-35-27b/pyTorch/default",
        f"{username}/qwen-35-4b/pyTorch/default",
    ]

    print("\n=== Checking Dataset Availability ===")
    for ds in datasets_to_check:
        try:
            path = kagglehub.dataset_download(ds)
            print(f"  [OK] {ds} → {path}")
        except Exception as e:
            print(f"  [MISSING] {ds}: {e}")

    print("\n=== Checking Model Availability ===")
    for model in models_to_check:
        try:
            path = kagglehub.model_download(model)
            print(f"  [OK] {model} → {path}")
        except Exception as e:
            print(f"  [MISSING] {model}: {e}")


def prepare_dataset_package(output_dir: Path, dataset_name: str, src_dir: Path):
    """Package a dataset directory for upload."""
    dest = output_dir / dataset_name
    dest.mkdir(parents=True, exist_ok=True)
    if src_dir.exists():
        for item in src_dir.iterdir():
            dst_item = dest / item.name
            if item.is_file():
                shutil.copy2(item, dst_item)
            elif item.is_dir():
                if dst_item.exists():
                    shutil.rmtree(dst_item)
                shutil.copytree(item, dst_item)
        file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
        print(f"  Prepared {dataset_name}: {file_count} files")
    else:
        print(f"  WARNING: Source not found: {src_dir}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Upload pipeline to Kaggle via kagglehub")
    parser.add_argument("--upload-code", action="store_true", help="Upload pipeline code")
    parser.add_argument("--upload-wheels", action="store_true", help="Upload Python wheels")
    parser.add_argument("--upload-data", action="store_true", help="Upload FinQA + ViNumQA datasets")
    parser.add_argument("--upload-models", action="store_true", help="Upload teacher + student model weights")
    parser.add_argument("--check", action="store_true", help="Check if uploads exist")
    parser.add_argument("--all", action="store_true", help="Upload everything (code + wheels + data)")
    parser.add_argument("--output-dir", default="/tmp/kaggle_upload", help="Temp directory")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing downloaded models (e.g., ./kaggle_offline_package/models)")
    args = parser.parse_args()

    has_action = args.upload_code or args.upload_wheels or args.upload_data or args.upload_models or args.check or args.all
    if not has_action:
        parser.print_help()
        return

    username = load_credentials()
    output_base = Path(args.output_dir)
    project_root = Path(__file__).parent.parent

    if args.check:
        check_uploads(username)
        if not (args.upload_code or args.upload_wheels or args.upload_data or args.upload_models or args.all):
            return

    if args.upload_code or args.all:
        print("\n=== Preparing Pipeline Code ===")
        code_dir = output_base / "vlsp2025-kd-pipeline"
        prepare_code_package(code_dir)
        upload_dataset_kagglehub(
            f"{username}/vlsp2025-kd-pipeline",
            str(code_dir),
            "Pipeline code with Qwen3.5 support",
        )

    if args.upload_wheels or args.all:
        print("\n=== Preparing Python Wheels ===")
        wheels_dir = output_base / "vlsp2025-kd-wheels"
        download_wheels(wheels_dir)
        upload_dataset_kagglehub(
            f"{username}/vlsp2025-kd-wheels",
            str(wheels_dir),
            "Offline Python dependencies",
        )

    if args.upload_data or args.all:
        print("\n=== Uploading Datasets ===")

        # FinQA English
        finqa_src = project_root / "dataset" / "finqa_en"
        if finqa_src.exists():
            finqa_dir = output_base / "finqa-en"
            prepare_dataset_package(output_base, "finqa-en", finqa_src)
            upload_dataset_kagglehub(
                f"{username}/finqa-en",
                str(finqa_dir),
                "FinQA English dataset",
            )

        # ViNumQA Private
        vinumqa_src = project_root / "dataset" / "viNumericalQA_private"
        if vinumqa_src.exists():
            vinumqa_dir = output_base / "vinumericalqa-private"
            prepare_dataset_package(output_base, "vinumericalqa-private", vinumqa_src)
            upload_dataset_kagglehub(
                f"{username}/vinumericalqa-private",
                str(vinumqa_dir),
                "ViNumQA Vietnamese private dataset",
            )

    if args.upload_models:
        print("\n=== Uploading Models ===")
        model_base = Path(args.model_dir) if args.model_dir else project_root / "kaggle_offline_package" / "models"
        if not model_base.exists():
            print(f"  ERROR: Model directory not found: {model_base}")
            print("  Download models first with: bash scripts/prepare_kaggle_offline.sh")
            print("  Or specify --model-dir /path/to/models")
        else:
            # Upload teacher model (Qwen3.5-27B)
            teacher_dir = model_base / "Qwen_Qwen3.5-27B"
            if teacher_dir.exists():
                upload_model_kagglehub(
                    f"{username}/qwen-35-27b/pyTorch/default",
                    str(teacher_dir),
                    "apache-2.0",
                    "Qwen3.5-27B teacher model",
                )
            else:
                print(f"  SKIP: Teacher model not found at {teacher_dir}")

            # Upload student model (Qwen3.5-4B)
            student_dir = model_base / "Qwen_Qwen3.5-4B"
            if student_dir.exists():
                upload_model_kagglehub(
                    f"{username}/qwen-35-4b/pyTorch/default",
                    str(student_dir),
                    "apache-2.0",
                    "Qwen3.5-4B student model",
                )
            else:
                print(f"  SKIP: Student model not found at {student_dir}")

    print("\n=== Summary ===")
    print(f"Kaggle username: {username}")
    print(f"\nIn Kaggle notebook, add these as input datasets:")
    print(f"  1. {username}/vlsp2025-kd-pipeline        (pipeline code)")
    print(f"  2. {username}/vlsp2025-kd-wheels           (offline wheels)")
    print(f"  3. {username}/finqa-en                     (FinQA dataset)")
    print(f"  4. {username}/vinumericalqa-private         (ViNumQA dataset)")
    print(f"\nAnd these as input models:")
    print(f"  5. {username}/qwen-35-27b/pyTorch/default  (teacher model)")
    print(f"  6. {username}/qwen-35-4b/pyTorch/default   (student model)")
    print(f"\nAlternatively, add HuggingFace models directly via Kaggle Model hub:")
    print(f"  5. Qwen/Qwen3.5-27B  (search 'Add Model' in Kaggle notebook)")
    print(f"  6. Qwen/Qwen3.5-4B   (search 'Add Model' in Kaggle notebook)")


if __name__ == "__main__":
    main()
