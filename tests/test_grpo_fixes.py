"""Tests for GRPO monkey-patches."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trl import GRPOTrainer

from src.training.grpo_fixes import apply_grpo_fixes


def _restore_grpo_trainer():
    """Best-effort restore of original methods after patching tests."""
    # Reloading the class from trl gives us a fresh GRPOTrainer.
    import importlib
    import trl.trainer.grpo_trainer as grpo_module
    importlib.reload(grpo_module)
    return grpo_module.GRPOTrainer


def test_apply_grpo_fixes_is_idempotent():
    """Calling apply_grpo_fixes multiple times must not nest wrappers."""
    # Work on a fresh class object to avoid side effects.
    FreshGRPOTrainer = _restore_grpo_trainer()

    orig_method = FreshGRPOTrainer._prepare_inputs
    apply_grpo_fixes(FreshGRPOTrainer)
    patched_once = FreshGRPOTrainer._prepare_inputs
    assert patched_once is not orig_method

    apply_grpo_fixes(FreshGRPOTrainer)
    patched_twice = FreshGRPOTrainer._prepare_inputs
    assert patched_twice is patched_once, "Second call should not re-wrap the method"
    print("test_apply_grpo_fixes_is_idempotent PASSED")


def _dummy_get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
    """Dummy original that just echoes input_ids for the wrapper to return."""
    return input_ids, None


def test_strip_image_pad_tokens_clones_input_ids():
    """The orphan-image-pad patch must not mutate the caller's input_ids tensor."""
    FreshGRPOTrainer = _restore_grpo_trainer()
    # Replace the real implementation with a dummy so the wrapper's truncation
    # logic runs but we don't need a full model forward.
    FreshGRPOTrainer._get_per_token_logps_and_entropies = _dummy_get_per_token_logps_and_entropies
    apply_grpo_fixes(FreshGRPOTrainer)

    # Fake trainer instance with the attributes the patch needs.
    class DummyTrainer:
        _is_vlm = True
        _image_pad_token_id = 99
        _video_pad_token_id = None

    # Build an input_ids tensor where the completion region contains an image pad.
    # prompt_len = 3, logits_to_keep = 4
    input_ids = torch.tensor([[1, 2, 3, 10, 99, 20, 30]])
    original = input_ids.clone()

    # Call the patched helper directly.
    trainer = DummyTrainer()
    method = FreshGRPOTrainer._get_per_token_logps_and_entropies
    returned, _ = method(
        trainer,
        model=None,
        input_ids=input_ids,
        attention_mask=None,
        logits_to_keep=4,
    )

    # The caller's tensor must be unchanged.
    assert torch.equal(input_ids, original), (
        f"input_ids was mutated in-place. Got {input_ids.tolist()}, expected {original.tolist()}"
    )
    # The returned tensor should be the truncated clone (image pad and after zeroed).
    expected_returned = torch.tensor([[1, 2, 3, 10, 0, 0, 0]])
    assert torch.equal(returned, expected_returned), (
        f"Truncated clone incorrect. Got {returned.tolist()}, expected {expected_returned.tolist()}"
    )
    print("test_strip_image_pad_tokens_clones_input_ids PASSED")


if __name__ == "__main__":
    test_apply_grpo_fixes_is_idempotent()
    test_strip_image_pad_tokens_clones_input_ids()
    print("\n=== GRPO fixes tests PASSED ===")
