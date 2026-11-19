import torch

from lora_behavior_steering.directions import (
    endpoint_pca_direction,
    mean_paired_difference,
    uncentered_difference_pca,
)
from lora_behavior_steering.effective_updates import inner_product


def update(values):
    return {"layer": torch.tensor(values, dtype=torch.float32).reshape(1, -1)}


def cosine(left, right):
    return inner_product(left, right) / (
        inner_product(left, left) ** 0.5 * inner_product(right, right) ** 0.5
    )


def test_shared_direction_estimators_recover_behavior_axis():
    low = [update([0, 0.1, 0]), update([0, 0, 0.1]), update([0, -0.1, -0.1])]
    high = [update([2, 0.1, 0]), update([2, 0, 0.1]), update([2, -0.1, -0.1])]
    expected = update([1, 0, 0])

    mean_direction = mean_paired_difference(low, high)
    svd_direction = uncentered_difference_pca(low, high)
    pca_result = endpoint_pca_direction(low, high)

    assert cosine(mean_direction, expected) > 0.999
    assert cosine(svd_direction, expected) > 0.999
    assert cosine(pca_result.direction, expected) > 0.999
    assert pca_result.paired_positive_rate == 1.0


def test_endpoint_pca_orients_low_to_high():
    low = [update([1, 0]), update([1, 1]), update([1, -1])]
    high = [update([-1, 0]), update([-1, 1]), update([-1, -1])]
    result = endpoint_pca_direction(low, high)
    paired_mean = mean_paired_difference(low, high)
    assert cosine(result.direction, paired_mean) > 0.999

