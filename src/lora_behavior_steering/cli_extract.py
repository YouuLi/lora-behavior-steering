from __future__ import annotations

import argparse
import json
from pathlib import Path

from .directions import endpoint_pca_direction, mean_paired_difference, uncentered_difference_pca
from .effective_updates import load_effective_update, save_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a behavior direction from paired LoRA endpoints.")
    parser.add_argument("--manifest", required=True, help="Endpoint manifest JSON file")
    parser.add_argument("--axis", required=True, choices=["detail", "register"])
    parser.add_argument(
        "--method",
        default="mean_difference",
        choices=["mean_difference", "uncentered_difference_pca", "endpoint_pca"],
    )
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--metadata", help="Optional output metadata JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    axis = manifest["axes"][args.axis]
    seeds = [str(seed) for seed in manifest["seeds"]]
    low_updates = [load_effective_update(axis["low"][seed]) for seed in seeds]
    high_updates = [load_effective_update(axis["high"][seed]) for seed in seeds]

    metadata: dict[str, object] = {
        "axis": args.axis,
        "method": args.method,
        "seeds": seeds,
        "low_label": axis.get("low_label", "low"),
        "high_label": axis.get("high_label", "high"),
    }
    if args.method == "mean_difference":
        direction = mean_paired_difference(low_updates, high_updates)
    elif args.method == "uncentered_difference_pca":
        direction = uncentered_difference_pca(low_updates, high_updates)
    else:
        result = endpoint_pca_direction(low_updates, high_updates)
        direction = result.direction
        metadata.update(
            {
                "selected_component": result.selected_component,
                "explained_variance_ratio": result.explained_variance_ratio,
                "standardized_separation": result.standardized_separation,
                "paired_positive_rate": result.paired_positive_rate,
                "used_consistency_fallback": result.used_consistency_fallback,
                "component_statistics": result.component_statistics,
            }
        )

    save_update(direction, args.output)
    metadata_path = Path(args.metadata) if args.metadata else Path(args.output).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved direction: {args.output}")
    print(f"Saved metadata:  {metadata_path}")


if __name__ == "__main__":
    main()

