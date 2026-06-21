"""Regression tests for filter_normal_level_data difficulty filtering."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.difficulty import filter_normal_level_data, is_rollout_correct


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


def test_filter_reward_threshold_mode(fake_setup):
    """When reward_threshold is set, correctness is determined by total_reward."""
    model, processor, samples = fake_setup

    # Return alternating rewards: 0.8, 0.2, 0.8, 0.2 for sample 0
    # and 0.1, 0.1, 0.1, 0.1 for sample 1 (all hard).
    call_count = [0]
    rewards = [0.8, 0.2, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1]

    def mock_compute_reward(**kwargs):
        r = rewards[call_count[0] % len(rewards)]
        call_count[0] += 1
        return {"total_reward": r}

    with (
        patch("src.utils.difficulty.batch_generate_completions", side_effect=_fake_batch_generate),
        patch("src.utils.difficulty.generate_single_completion"),
        patch("src.utils.difficulty.compute_total_reward", side_effect=mock_compute_reward),
    ):
        result = filter_normal_level_data(
            model=model,
            processor=processor,
            data=samples,
            num_generations=4,
            batch_size=2,
            reward_threshold=0.5,
        )

    # Sample 0: rewards [0.8, 0.2, 0.8, 0.2] with threshold 0.5
    #   -> correct_count = 2 (0.8 >= 0.5, 0.2 < 0.5, 0.8 >= 0.5, 0.2 < 0.5) -> Normal
    # Sample 1: rewards [0.1, 0.1, 0.1, 0.1] with threshold 0.5
    #   -> correct_count = 0 -> Hard
    assert len(result) == 1


# ── Tests for is_rollout_correct IoU/distance-based logic ────────────────────

_THINK_WRAP = "<think>some reasoning</think>\n"
_BOX_OPEN = "<|box|>"
_BOX_CLOSE = "<|/box|>"


def _make_box_text(boxes):
    """Build a minimal valid box output with think tags."""
    box_str = "".join(f"{_BOX_OPEN}[[{','.join(map(str,b))}]]{_BOX_CLOSE}" for b in boxes)
    return _THINK_WRAP + box_str


def test_box_iou_above_threshold_is_correct():
    pred = _make_box_text([[100, 100, 300, 300]])
    gt = _make_box_text([[100, 100, 300, 300]])  # IoU = 1.0
    assert is_rollout_correct(pred, gt, task_type="box", iou_threshold=0.3)


def test_box_iou_below_threshold_is_incorrect():
    # Pred box far from GT -> IoU ≈ 0
    pred = _make_box_text([[10, 10, 30, 30]])
    gt = _make_box_text([[500, 500, 700, 700]])
    assert not is_rollout_correct(pred, gt, task_type="box", iou_threshold=0.3)


def test_box_partial_overlap_above_threshold():
    # GT: [100,100,300,300] (200×200=40000)
    # Pred: [150,100,350,300] → intersection = [150,100,300,300] = 150×200 = 30000
    # union = 40000 + 40000 - 30000 = 50000, IoU = 30000/50000 = 0.6
    pred = _make_box_text([[150, 100, 350, 300]])
    gt = _make_box_text([[100, 100, 300, 300]])
    assert is_rollout_correct(pred, gt, task_type="box", iou_threshold=0.5)


def test_box_no_gt_boxes_falls_back_to_answer_match():
    # Counting task: GT has no boxes, answer is a number.
    pred = _THINK_WRAP + "The answer is 3."
    gt = "<think>counting</think>\nThe answer is 3."
    assert is_rollout_correct(pred, gt, task_type="box", iou_threshold=0.3)


def test_box_no_gt_boxes_answer_mismatch_is_incorrect():
    pred = _THINK_WRAP + "The answer is 3."
    gt = "<think>counting</think>\nThe answer is 5."
    assert not is_rollout_correct(pred, gt, task_type="box", iou_threshold=0.3)


def test_box_no_think_tags_still_correct_if_iou_matches():
    # Think tags are a FORMAT concern, not a task-correctness concern.
    # Paper's "correct" = "model solved the task" (IoU ≥ threshold),
    # not "model produced perfect format".
    pred = "<|box|>[[100,100,300,300]]<|/box|>"
    gt = _make_box_text([[100, 100, 300, 300]])
    assert is_rollout_correct(pred, gt, task_type="box", iou_threshold=0.3)


def test_point_within_threshold_is_correct():
    _POINT_OPEN = "<|point|>"
    _POINT_CLOSE = "<|/point|>"
    pred = _THINK_WRAP + f"{_POINT_OPEN}[[100,100]]{_POINT_CLOSE}"
    gt = _THINK_WRAP + f"{_POINT_OPEN}[[105,105]]{_POINT_CLOSE}"
    assert is_rollout_correct(pred, gt, task_type="point", point_dist_threshold=20.0)


def test_point_beyond_threshold_is_incorrect():
    _POINT_OPEN = "<|point|>"
    _POINT_CLOSE = "<|/point|>"
    pred = _THINK_WRAP + f"{_POINT_OPEN}[[100,100]]{_POINT_CLOSE}"
    gt = _THINK_WRAP + f"{_POINT_OPEN}[[500,500]]{_POINT_CLOSE}"
    assert not is_rollout_correct(pred, gt, task_type="point", point_dist_threshold=20.0)
