"""Download public benchmark splits into a local JSON cache for offline runs.

This is primarily used to prepare Kaggle-friendly benchmark mirrors for the
main reasoning suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import load_config


def _save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def build_cache(benchmarks: list[str], gpu_profile: str = "p100_16gb", config_path: str | None = None) -> None:
    cfg = load_config(gpu_profile=gpu_profile, config_path=config_path)

    for benchmark in benchmarks:
        source = cfg.data.benchmark_sources.get(benchmark)
        if source is None:
            raise KeyError(f"Unknown benchmark cache target: {benchmark}")

        cache_dir = source.get("cache_dir")
        if not cache_dir:
            print(f"Skipping {benchmark}: no cache_dir configured")
            continue

        split_map = source.get("split_map", {})
        output_dir = Path(cfg.project_root) / cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for purpose, split_name in split_map.items():
            try:
                if benchmark == "convfinqa":
                    dataset = load_dataset(source["dataset_name"], source.get("config_name"))
                    rows = list(dataset[split_name])
                else:
                    dataset = load_dataset(
                        source["dataset_name"],
                        source.get("config_name"),
                        split=split_name,
                    )
                    rows = list(dataset)
            except Exception as exc:
                print(f"Skipping {benchmark}:{purpose} ({split_name}) -> {exc}")
                continue

            filename = source.get("local_splits", {}).get(purpose, f"{split_name}.json")
            output_path = output_dir / filename
            _save_rows(output_path, rows)
            print(f"Saved {benchmark}:{purpose} -> {output_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local benchmark caches for offline runs")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["tatqa", "convfinqa"],
        help="Benchmarks to cache locally",
    )
    parser.add_argument("--gpu-profile", default="p100_16gb")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    build_cache(args.benchmarks, gpu_profile=args.gpu_profile, config_path=args.config)


if __name__ == "__main__":
    main()