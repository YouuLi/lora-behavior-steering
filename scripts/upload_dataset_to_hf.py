from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

FILES = (
    "alpaca_detail_score1_1k.jsonl",
    "alpaca_detail_score5_1k.jsonl",
    "alpaca_register_score1_1k.jsonl",
    "alpaca_register_score5_1k.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload behavior endpoint data to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="account-or-org/dataset-name")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--card", type=Path, default=Path("hf_dataset/README.md"))
    args = parser.parse_args()

    missing = [name for name in FILES if not (args.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")
    if not args.card.exists():
        raise FileNotFoundError(f"Dataset card not found: {args.card}")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    for name in FILES:
        api.upload_file(
            path_or_fileobj=str(args.data_dir / name),
            path_in_repo=f"data/{name}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {name}",
        )
        print(f"Uploaded {name}")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

