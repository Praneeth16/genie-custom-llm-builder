import pytest

from train_grpo import validate_generation_batch_size


def test_generation_batch_size_accepts_divisible_configuration():
    validate_generation_batch_size(8, 4, 8)


def test_generation_batch_size_rejects_non_divisible_configuration():
    with pytest.raises(ValueError, match=r"per_device_train_batch_size \* gradient_accumulation_steps \(4\) must be divisible by num_generations \(8\)"):
        validate_generation_batch_size(1, 4, 8)
