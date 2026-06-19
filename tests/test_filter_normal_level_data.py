"""Regression tests for filter_normal_level_data difficulty filtering."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.difficulty import filter_normal_level_data


class FakeModel:
    """Minimal model stand-in for filter_normal_level_data."""

    def __init__(self):
        self.training = True
        self.is_gradient_checkpointing = False
        self.gradient_checkpointing = False

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def gradient_checkpointing_disable(self):
        pass

    def gradient_checkpointing_enable(self):
        pass

    def modules(self):
        return [self]

    def children(self):
        return []


@pytest.fixture
def fake_setup():
    model = FakeModel()
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.tokenizer.eos_token_id = 1
    processor.device = torch.device("cpu")

    def decode(ids, skip_special_tokens=False):
        # Return a deterministic string based on the first token.
        token = ids[0].item() if hasattr(ids, "tolist") else ids[0]
        return f"pred_{token}"

    processor.tokenizer.decode = decode

    samples = [
        {"prompt": "p1", "reasoning": "r1", "answer": "1"},
        {"prompt": "p2", "reasoning": "r2", "answer": "2"},
    ]
    return model, processor, samples


def _fake_batch_generate(model, processor, samples, num_generations, **kwargs):
    # outputs shape: (len(samples) * num_generations, seq_len)
    seq_len = 10
    total = len(samples) * num_generations
    outputs = torch.arange(total).unsqueeze(1).repeat(1, seq_len)
    input_len = 5
    return outputs, input_len


def _fake_is_rollout_correct(pred_text, gt_text, **kwargs):
    # Mark every even-indexed rollout correct, odd incorrect -> Normal kept.
    token = int(pred_text.split("_")[1])
    return token % 2 == 0


def test_filter_keeps_normal_samples(fake_setup):
    model, processor, samples = fake_setup

    with (
        patch("src.utils.difficulty.batch_generate_completions", side_effect=_fake_batch_generate),
        patch("src.utils.difficulty.generate_single_completion") as single_mock,
        patch("src.utils.difficulty.compute_total_reward", return_value={"total_reward": 1.0}),
        patch("src.utils.difficulty.is_rollout_correct", side_effect=_fake_is_rollout_correct),
    ):
        result = filter_normal_level_data(
            model=model,
            processor=processor,
            data=samples,
            num_generations=4,
            batch_size=2,
        )

    # With 4 generations and even/odd correctness, every sample has both correct
    # and incorrect rollouts -> Normal.
    assert len(result) == 2
    # Fallback singles should never be hit.
    single_mock.assert_not_called()


def test_filter_skips_easy_and_hard(fake_setup):
    model, processor, samples = fake_setup

    def all_correct(pred_text, gt_text, **kwargs):
        return True

    with (
        patch("src.utils.difficulty.batch_generate_completions", side_effect=_fake_batch_generate),
        patch("src.utils.difficulty.generate_single_completion") as single_mock,
        patch("src.utils.difficulty.compute_total_reward", return_value={"total_reward": 1.0}),
        patch("src.utils.difficulty.is_rollout_correct", side_effect=all_correct),
    ):
        result = filter_normal_level_data(
            model=model,
            processor=processor,
            data=samples,
            num_generations=4,
            batch_size=2,
        )

    assert len(result) == 0
    single_mock.assert_not_called()
