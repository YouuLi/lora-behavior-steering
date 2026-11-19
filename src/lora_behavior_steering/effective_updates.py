from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file

Update = dict[str, torch.Tensor]


def load_effective_update(adapter_dir: str | Path) -> Update:
    """Load a PEFT LoRA adapter and return scaled dense updates s * B @ A."""
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = float(config["lora_alpha"]) / float(config["r"])
    state = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    prefixes = sorted({key.split(".lora_A.")[0] for key in state if ".lora_A." in key})
    if not prefixes:
        raise ValueError(f"No LoRA matrices found in {adapter_dir}")

    update: Update = {}
    for prefix in prefixes:
        a_keys = [key for key in state if key.startswith(prefix + ".lora_A.")]
        b_keys = [key for key in state if key.startswith(prefix + ".lora_B.")]
        if len(a_keys) != 1 or len(b_keys) != 1:
            raise ValueError(f"Ambiguous LoRA tensors for {prefix}")
        update[prefix] = (state[b_keys[0]].float() @ state[a_keys[0]].float()) * scale
    return update


def validate_compatible(updates: list[Mapping[str, torch.Tensor]]) -> list[str]:
    if not updates:
        raise ValueError("At least one update is required")
    keys = sorted(updates[0])
    reference = {key: tuple(updates[0][key].shape) for key in keys}
    for index, update in enumerate(updates[1:], start=1):
        if sorted(update) != keys:
            raise ValueError(f"Update {index} has a different module set")
        for key in keys:
            if tuple(update[key].shape) != reference[key]:
                raise ValueError(f"Shape mismatch for {key} in update {index}")
    return keys


def inner_product(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if sorted(left) != sorted(right):
        raise ValueError("Updates have different module sets")
    return float(sum(torch.sum(left[key].double() * right[key].double()).item() for key in left))


def norm(update: Mapping[str, torch.Tensor]) -> float:
    return max(inner_product(update, update), 0.0) ** 0.5


def linear_combination(
    updates: list[Mapping[str, torch.Tensor]], coefficients: list[float]
) -> Update:
    if len(updates) != len(coefficients):
        raise ValueError("Number of updates and coefficients must match")
    keys = validate_compatible(updates)
    return {
        key: sum(float(c) * update[key].float() for update, c in zip(updates, coefficients))
        for key in keys
    }


def normalize(update: Mapping[str, torch.Tensor], eps: float = 1e-12) -> Update:
    length = norm(update)
    if length <= eps:
        raise ValueError("Cannot normalize a zero update")
    return {key: value.float() / length for key, value in update.items()}


def save_update(update: Mapping[str, torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous().cpu() for key, value in update.items()}, str(path))

