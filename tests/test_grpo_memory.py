"""Tests for GRPO memory optimizations."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.training.memory_utils import cast_ref_adapter_to_bf16
from src.training.callbacks import CastRefAdapterCallback


class _MockParam:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad


class _MockModel:
    def __init__(self, has_ref=True):
        self.peft_config = {"default": None}
        if has_ref:
            self.peft_config["ref"] = None
        self._params = {
            "base_model.model.lm_head.weight": _MockParam(torch.ones(10, 10, dtype=torch.float32)),
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": _MockParam(
                torch.ones(5, 5, dtype=torch.float32), requires_grad=True
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_A.ref.weight": _MockParam(
                torch.ones(5, 5, dtype=torch.float32)
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": _MockParam(
                torch.ones(5, 5, dtype=torch.float32), requires_grad=True
            ),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.ref.weight": _MockParam(
                torch.ones(5, 5, dtype=torch.float32)
            ),
        }

    def named_parameters(self):
        return self._params.items()


def test_cast_ref_adapter_to_bf16_casts_only_ref():
    model = _MockModel(has_ref=True)
    count = cast_ref_adapter_to_bf16(model)
    assert count == 2

    for name, param in model.named_parameters():
        if ".ref." in name:
            assert param.data.dtype == torch.bfloat16, f"{name} should be bf16"
        else:
            assert param.data.dtype == torch.float32, f"{name} should stay fp32"
    print("test_cast_ref_adapter_to_bf16_casts_only_ref PASSED")


def test_cast_ref_adapter_to_bf16_no_ref_is_noop():
    model = _MockModel(has_ref=False)
    count = cast_ref_adapter_to_bf16(model)
    assert count == 0
    for _, param in model.named_parameters():
        assert param.data.dtype == torch.float32
    print("test_cast_ref_adapter_to_bf16_no_ref_is_noop PASSED")


def test_cast_ref_adapter_callback_runs_helper():
    model = _MockModel(has_ref=True)
    callback = CastRefAdapterCallback()
    callback.on_train_begin(args=None, state=None, control=None, model=model)

    for name, param in model.named_parameters():
        if ".ref." in name:
            assert param.data.dtype == torch.bfloat16
    print("test_cast_ref_adapter_callback_runs_helper PASSED")


def test_cast_ref_adapter_callback_handles_missing_model():
    callback = CastRefAdapterCallback()
    # Should not raise when model is None
    callback.on_train_begin(args=None, state=None, control=None, model=None)
    print("test_cast_ref_adapter_callback_handles_missing_model PASSED")


if __name__ == "__main__":
    test_cast_ref_adapter_to_bf16_casts_only_ref()
    test_cast_ref_adapter_to_bf16_no_ref_is_noop()
    test_cast_ref_adapter_callback_runs_helper()
    test_cast_ref_adapter_callback_handles_missing_model()
    print("\n=== GRPO memory tests PASSED ===")
