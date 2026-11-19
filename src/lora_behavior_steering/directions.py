from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .effective_updates import (
    Update,
    inner_product,
    linear_combination,
    normalize,
    validate_compatible,
)


def _subtract(high: Mapping[str, torch.Tensor], low: Mapping[str, torch.Tensor]) -> Update:
    validate_compatible([low, high])
    return {key: high[key].float() - low[key].float() for key in low}


def mean_paired_difference(
    low_updates: Sequence[Mapping[str, torch.Tensor]],
    high_updates: Sequence[Mapping[str, torch.Tensor]],
) -> Update:
    """Normalized mean of paired high-minus-low effective updates.

    This is the recommended estimator of a shared cross-seed behavior direction.
    It preserves the common mean contrast that centered PCA on differences removes.
    """
    if len(low_updates) != len(high_updates) or not low_updates:
        raise ValueError("Low and high endpoints must be non-empty paired lists")
    differences = [_subtract(high, low) for low, high in zip(low_updates, high_updates)]
    return normalize(linear_combination(differences, [1.0 / len(differences)] * len(differences)))


def uncentered_difference_pca(
    low_updates: Sequence[Mapping[str, torch.Tensor]],
    high_updates: Sequence[Mapping[str, torch.Tensor]],
) -> Update:
    """Leading right singular direction of uncentered paired differences."""
    if len(low_updates) != len(high_updates) or not low_updates:
        raise ValueError("Low and high endpoints must be non-empty paired lists")
    differences = [_subtract(high, low) for low, high in zip(low_updates, high_updates)]
    gram = _gram(differences)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    index = int(np.argmax(eigenvalues))
    if eigenvalues[index] <= 0:
        raise ValueError("Difference matrix has no positive singular value")
    coefficients = eigenvectors[:, index] / np.sqrt(eigenvalues[index])
    direction = normalize(linear_combination(differences, coefficients.tolist()))
    mean_direction = mean_paired_difference(low_updates, high_updates)
    if inner_product(direction, mean_direction) < 0:
        direction = {key: -value for key, value in direction.items()}
    return direction


@dataclass(frozen=True)
class EndpointPCAResult:
    direction: Update
    selected_component: int
    explained_variance_ratio: float
    standardized_separation: float
    paired_positive_rate: float
    used_consistency_fallback: bool
    component_statistics: list[dict[str, float | int | list[float]]]


def endpoint_pca_direction(
    low_updates: Sequence[Mapping[str, torch.Tensor]],
    high_updates: Sequence[Mapping[str, torch.Tensor]],
    *,
    min_explained_variance: float = 0.01,
) -> EndpointPCAResult:
    """Reproduce the paper's pooled-endpoint PCA and labeled PC selection.

    PCA is performed on all centered endpoints. Components must explain at least
    ``min_explained_variance`` and, when possible, order every paired high
    endpoint on the same side of its low endpoint. The retained component with
    the largest absolute standardized group separation is selected.
    """
    if len(low_updates) != len(high_updates) or not low_updates:
        raise ValueError("Low and high endpoints must be non-empty paired lists")

    endpoints: list[Mapping[str, torch.Tensor]] = []
    labels: list[int] = []
    for low, high in zip(low_updates, high_updates):
        validate_compatible([low, high])
        endpoints.extend([low, high])
        labels.extend([0, 1])

    mean_update = linear_combination(endpoints, [1.0 / len(endpoints)] * len(endpoints))
    centered = [_subtract(endpoint, mean_update) for endpoint in endpoints]
    gram = _gram(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    threshold = max(float(eigenvalues[0]), 1.0) * 1e-10
    keep = eigenvalues > threshold
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    if not len(eigenvalues):
        raise ValueError("Centered endpoints have no positive variance")

    scores = eigenvectors * np.sqrt(eigenvalues)[None, :]
    explained = eigenvalues / eigenvalues.sum()
    labels_array = np.asarray(labels)
    stats: list[dict[str, float | int | list[float]]] = []
    for component in range(scores.shape[1]):
        low_scores = scores[labels_array == 0, component]
        high_scores = scores[labels_array == 1, component]
        paired = high_scores - low_scores
        pooled_sd = np.sqrt((np.var(low_scores, ddof=1) + np.var(high_scores, ddof=1)) / 2)
        separation = float((high_scores.mean() - low_scores.mean()) / max(pooled_sd, 1e-12))
        stats.append(
            {
                "component": component + 1,
                "explained_variance_ratio": float(explained[component]),
                "standardized_separation": separation,
                "paired_positive_rate": float(np.mean(paired > 0)),
                "paired_differences": paired.tolist(),
            }
        )

    eligible = [
        row
        for row in stats
        if row["explained_variance_ratio"] >= min_explained_variance
        and row["paired_positive_rate"] in (0.0, 1.0)
    ]
    fallback = not eligible
    if fallback:
        eligible = [row for row in stats if row["explained_variance_ratio"] >= min_explained_variance]
    if not eligible:
        raise ValueError("No PCA component meets the explained-variance threshold")

    selected_row = max(eligible, key=lambda row: abs(float(row["standardized_separation"])))
    selected = int(selected_row["component"]) - 1
    coefficients = eigenvectors[:, selected] / np.sqrt(eigenvalues[selected])
    direction = normalize(linear_combination(centered, coefficients.tolist()))
    if float(selected_row["standardized_separation"]) < 0:
        direction = {key: -value for key, value in direction.items()}
        selected_row["standardized_separation"] = -float(selected_row["standardized_separation"])
        selected_row["paired_positive_rate"] = float(
            np.mean(np.asarray(selected_row["paired_differences"]) < 0)
        )
        selected_row["paired_differences"] = [
            -float(value) for value in selected_row["paired_differences"]
        ]

    return EndpointPCAResult(
        direction=direction,
        selected_component=selected + 1,
        explained_variance_ratio=float(explained[selected]),
        standardized_separation=float(selected_row["standardized_separation"]),
        paired_positive_rate=float(selected_row["paired_positive_rate"]),
        used_consistency_fallback=fallback,
        component_statistics=stats,
    )


def _gram(updates: Sequence[Mapping[str, torch.Tensor]]) -> np.ndarray:
    validate_compatible(list(updates))
    gram = np.zeros((len(updates), len(updates)), dtype=np.float64)
    for left in range(len(updates)):
        for right in range(left, len(updates)):
            value = inner_product(updates[left], updates[right])
            gram[left, right] = value
            gram[right, left] = value
    return gram

