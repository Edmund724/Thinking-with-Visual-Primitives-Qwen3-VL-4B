"""Integration test: verify reward function receives data correctly from GRPO kwargs
and handles conversational completion format + skip_special_tokens re-decoding."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_stage4b_grpo_point import make_point_reward_fn
from scripts.run_stage4a_grpo_box import make_box_reward_fn
from src.data.datasets.grpo_dataset import _build_gt_text
from src.training.grpo_utils import extract_completion_text
from transformers import AutoProcessor
from src.utils.constants import SPECIAL_TOKENS

# Load tokenizer for re-decoding test
proc = AutoProcessor.from_pretrained("outputs/stage3b_sft_point", trust_remote_code=True)
proc.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
tokenizer = proc.tokenizer


def test_extract_completion_text_conversational():
    """Verify extract_completion_text handles conversational message list."""
    msg = [{"role": "assistant", "content": "I see a cat. The answer is 2."}]
    text = extract_completion_text(msg)
    assert text == "I see a cat. The answer is 2.", f"Got: {text!r}"
    print("test_extract_completion_text_conversational PASSED")


def test_extract_completion_text_from_ids():
    """Verify re-decoding from completion_ids preserves special tokens."""
    LT, GT, PIPE = chr(60), chr(62), chr(124)
    box_open = f"{LT}{PIPE}box{PIPE}{GT}"
    box_close = f"{LT}{PIPE}/box{PIPE}{GT}"
    original = f"I see. {LT}think{GT}thinking{LT}/think{GT} obj at {box_open}[[100,100,200,200]]{box_close} done"
    ids = tokenizer.encode(original, add_special_tokens=False)
    # Add EOS token like TRL does
    ids_with_eos = ids + [tokenizer.eos_token_id]
    text = extract_completion_text(None, tokenizer=tokenizer, completion_id=ids_with_eos)
    assert box_open in text, f"box_open missing from: {text!r}"
    assert box_close in text, f"box_close missing from: {text!r}"
    assert tokenizer.eos_token not in text, f"EOS should be stripped: {text!r}"
    print(f"Re-decoded: {text!r}")
    print("test_extract_completion_text_from_ids PASSED")


def test_point_reward_with_conversational_and_ids():
    """Verify point reward works with conversational completions + completion_ids."""
    reward_fn = make_point_reward_fn(point_dist_threshold=20.0, tokenizer=tokenizer)

    # Build a completion that the model might generate
    LT, GT, PIPE = chr(60), chr(62), chr(124)
    think_open = f"{LT}think{GT}"
    think_close = f"{LT}/think{GT}"
    point_open = f"{LT}{PIPE}point{PIPE}{GT}"
    point_close = f"{LT}{PIPE}/point{PIPE}{GT}"

    completion_text = f"Let me find it. {think_open}The object is at {point_open}[[500, 500]]{point_close}{think_close}\n\nThe answer is (500, 500)."
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    # TRL adds EOS
    completion_ids_with_eos = completion_ids + [tokenizer.eos_token_id]

    # TRL wraps in conversational format
    completions = [[{"role": "assistant", "content": "decoded_without_special_tokens"}]]

    sample = {
        "reasoning": f"{think_open}Intent: find object.\nGrounding: at {point_open}[[500, 500]]{point_close}\nSummary: found.{think_close}",
        "answer": "(500, 500)",
        "task_type": "point",
    }
    gt_text = _build_gt_text(sample)

    rewards = reward_fn(
        completions=completions,
        prompts=["prompt1"],
        gt_text=[gt_text],
        task_type=["point"],
        maze_grid=[None],
        completion_ids=[completion_ids_with_eos],
    )

    assert len(rewards) == 1
    assert rewards[0] > 0.0, f"Expected positive reward, got {rewards[0]}"
    print(f"Point reward with completion_ids: {rewards[0]}")
    print("test_point_reward_with_conversational_and_ids PASSED")


def test_box_reward_with_conversational_and_ids():
    """Verify box reward works with conversational completions + completion_ids."""
    reward_fn = make_box_reward_fn(iou_threshold=0.3, tokenizer=tokenizer)

    LT, GT, PIPE = chr(60), chr(62), chr(124)
    think_open = f"{LT}think{GT}"
    think_close = f"{LT}/think{GT}"
    box_open = f"{LT}{PIPE}box{PIPE}{GT}"
    box_close = f"{LT}{PIPE}/box{PIPE}{GT}"

    completion_text = f"{think_open}I see object at {box_open}[[100, 100, 200, 200]]{box_close}{think_close}\n\nThe answer is 1."
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    completion_ids_with_eos = completion_ids + [tokenizer.eos_token_id]

    completions = [[{"role": "assistant", "content": "dummy"}]]

    sample = {
        "reasoning": f"{think_open}Intent: find.\nGrounding: at {box_open}[[100, 100, 200, 200]]{box_close}\nSummary: 1 obj.{think_close}",
        "answer": "1",
        "task_type": "box",
    }
    gt_text = _build_gt_text(sample)

    rewards = reward_fn(
        completions=completions,
        prompts=["prompt1"],
        gt_text=[gt_text],
        completion_ids=[completion_ids_with_eos],
    )

    assert len(rewards) == 1
    assert rewards[0] > 0.0, f"Expected positive reward, got {rewards[0]}"
    print(f"Box reward with completion_ids: {rewards[0]}")
    print("test_box_reward_with_conversational_and_ids PASSED")


def test_empty_kwargs_returns_zeros():
    """Verify reward function gracefully handles missing kwargs."""
    reward_fn = make_point_reward_fn(point_dist_threshold=20.0, tokenizer=tokenizer)
    completions = [[{"role": "assistant", "content": "some text"}]]
    rewards = reward_fn(completions=completions)
    assert len(rewards) == 1
    assert rewards[0] == 0.0  # no gt_text → 0.0
    print("test_empty_kwargs_returns_zeros PASSED")


if __name__ == "__main__":
    test_extract_completion_text_conversational()
    test_extract_completion_text_from_ids()
    test_point_reward_with_conversational_and_ids()
    test_box_reward_with_conversational_and_ids()
    test_empty_kwargs_returns_zeros()
    print("\n=== All integration tests PASSED ===")
