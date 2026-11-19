from __future__ import annotations

import argparse
import json
from pathlib import Path

FILES = {
    "detail_target_1": "alpaca_detail_score1_1k.jsonl",
    "detail_target_5": "alpaca_detail_score5_1k.jsonl",
    "register_target_1": "alpaca_register_score1_1k.jsonl",
    "register_target_5": "alpaca_register_score5_1k.jsonl",
}
REQUIRED_FIELDS = {"instruction", "input", "output"}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if not isinstance(row["instruction"], str) or not row["instruction"].strip():
                raise ValueError(f"{path}:{line_number}: instruction must be non-empty text")
            if not isinstance(row["input"], str) or not isinstance(row["output"], str):
                raise ValueError(f"{path}:{line_number}: input and output must be strings")
            if not row["output"].strip():
                raise ValueError(f"{path}:{line_number}: output must be non-empty")
            rows.append(row)
    return rows


def pair_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["instruction"]), str(row["input"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the four Hugging Face endpoint files.")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    datasets = {name: read_jsonl(args.data_dir / filename) for name, filename in FILES.items()}
    for name, rows in datasets.items():
        keys = [pair_key(row) for row in rows]
        duplicates = len(keys) - len(set(keys))
        print(f"{name}: rows={len(rows)}, duplicate instruction/input pairs={duplicates}")

    for axis, low_name, high_name in (
        ("detail", "detail_target_1", "detail_target_5"),
        ("register", "register_target_1", "register_target_5"),
    ):
        low_keys = [pair_key(row) for row in datasets[low_name]]
        high_keys = [pair_key(row) for row in datasets[high_name]]
        if low_keys != high_keys:
            low_set, high_set = set(low_keys), set(high_keys)
            raise ValueError(
                f"{axis} endpoints are not positionally aligned; "
                f"only-low={len(low_set-high_set)}, only-high={len(high_set-low_set)}"
            )
        print(f"{axis}: paired alignment OK ({len(low_keys)} pairs)")


if __name__ == "__main__":
    main()

