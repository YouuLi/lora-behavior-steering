from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch


def _candidate_parameter_names(prefix: str) -> list[str]:
    names = [prefix + ".weight"]
    for leading in ("base_model.model.", "base_model."):
        if prefix.startswith(leading):
            names.append(prefix[len(leading) :] + ".weight")
    return list(dict.fromkeys(names))


def resolve_parameter_name(prefix: str, parameter_names: set[str]) -> str:
    for candidate in _candidate_parameter_names(prefix):
        if candidate in parameter_names:
            return candidate
    raise KeyError(f"Cannot map LoRA module {prefix!r} to a base-model weight")


@contextmanager
def temporary_steering(
    model: torch.nn.Module,
    directions: list[tuple[float, Mapping[str, torch.Tensor]]],
) -> Iterator[None]:
    """Temporarily add weighted dense directions to model parameters."""
    parameters = dict(model.named_parameters())
    names = set(parameters)
    applied: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for coefficient, direction in directions:
                if coefficient == 0:
                    continue
                for prefix, value in direction.items():
                    name = resolve_parameter_name(prefix, names)
                    parameter = parameters[name]
                    delta = value.to(device=parameter.device, dtype=parameter.dtype) * float(coefficient)
                    if parameter.shape != delta.shape:
                        raise ValueError(f"Shape mismatch for {name}: {parameter.shape} vs {delta.shape}")
                    parameter.add_(delta)
                    applied.append((parameter, delta))
        yield
    finally:
        with torch.no_grad():
            for parameter, delta in reversed(applied):
                parameter.sub_(delta)

