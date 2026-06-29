"""Tests for Qwen-VL model loader helpers."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.models.qwen_vl_loader import _patch_lm_head_dtype_cast


class _FakeModulesToSaveWrapper(nn.Module):
    """Minimal stand-in for PEFT's ModulesToSaveWrapper."""

    def __init__(self, linear, adapter="default"):
        super().__init__()
        self.modules_to_save = nn.ModuleDict({adapter: linear})
        self.active_adapters = [adapter]

    def forward(self, x, *args, **kwargs):
        return self.modules_to_save[self.active_adapters[0]](x, *args, **kwargs)


class _FakeModel(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head


def test_patch_lm_head_dtype_cast_with_peft_wrapper():
    """fp32 inputs to a bf16 lm_head must be cast before the linear call."""
    inner = nn.Linear(8, 4, dtype=torch.bfloat16)
    wrapper = _FakeModulesToSaveWrapper(inner)
    model = _FakeModel(wrapper)

    _patch_lm_head_dtype_cast(model)

    # Simulate fp32 hidden states as produced by layer norms during generation.
    fp32_input = torch.randn(2, 8, dtype=torch.float32)
    output = model.lm_head(fp32_input)

    assert output.dtype == torch.bfloat16, (
        f"Expected bf16 output, got {output.dtype}"
    )
    # The inner module should still have produced the same numeric result as a
    # manually casted call.
    expected = inner(fp32_input.to(torch.bfloat16))
    assert torch.allclose(output, expected)
    print("test_patch_lm_head_dtype_cast_with_peft_wrapper PASSED")


def test_patch_lm_head_dtype_cast_idempotent():
    """Calling the patch twice must not stack wrappers."""
    inner = nn.Linear(8, 4, dtype=torch.bfloat16)
    wrapper = _FakeModulesToSaveWrapper(inner)
    model = _FakeModel(wrapper)

    _patch_lm_head_dtype_cast(model)
    first_forward = inner.forward
    _patch_lm_head_dtype_cast(model)

    assert inner.forward is first_forward, "Patch should be idempotent"
    print("test_patch_lm_head_dtype_cast_idempotent PASSED")


def test_patch_lm_head_dtype_cast_no_wrapper():
    """Patch also works on a plain (non-PEFT-wrapped) lm_head."""
    inner = nn.Linear(8, 4, dtype=torch.bfloat16)
    model = _FakeModel(inner)

    _patch_lm_head_dtype_cast(model)

    fp32_input = torch.randn(2, 8, dtype=torch.float32)
    output = model.lm_head(fp32_input)
    assert output.dtype == torch.bfloat16
    print("test_patch_lm_head_dtype_cast_no_wrapper PASSED")


def test_patch_lm_head_dtype_cast_bf16_weight():
    """Reproduce the user's exact error case: fp32 input + bf16 weight."""
    inner = nn.Linear(8, 4, dtype=torch.bfloat16)
    wrapper = _FakeModulesToSaveWrapper(inner)
    model = _FakeModel(wrapper)

    _patch_lm_head_dtype_cast(model)

    fp32_input = torch.randn(2, 8, dtype=torch.float32)
    # This would raise RuntimeError without the patch.
    output = model.lm_head(fp32_input)
    assert output.dtype == torch.bfloat16
    print("test_patch_lm_head_dtype_cast_bf16_weight PASSED")


if __name__ == "__main__":
    test_patch_lm_head_dtype_cast_with_peft_wrapper()
    test_patch_lm_head_dtype_cast_idempotent()
    test_patch_lm_head_dtype_cast_no_wrapper()
    test_patch_lm_head_dtype_cast_bf16_weight()
    print("\n=== qwen_vl_loader tests PASSED ===")
