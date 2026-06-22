"""Regression test for stage3a SFT data composition.

The script used to reference `all_data` before assignment while adding
negative-box samples. This test mocks all heavy dependencies and verifies
that the script composes a training set containing every sample type,
including COCO negative boxes.
"""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "run_stage3a_sft_box.py"

spec = importlib.util.spec_from_file_location("run_stage3a_sft_box", SCRIPT_PATH)
stage3a = importlib.util.module_from_spec(spec)
sys.modules["run_stage3a_sft_box"] = stage3a
spec.loader.exec_module(stage3a)


def _make_args():
    return Namespace(
        model_path="dummy_model",
        output_dir="outputs/test_stage3a",
        general_data_path="data/pretrain/pretrain_data.json",
        coco_image_dir="data/coco",
        coco_ann_file="data/coco/ann.json",
        num_box=2,
        num_counting=2,
        counting_attribute_ratio=0.0,
        num_clevr=2,
        clevr_negative_ratio=0.0,
        num_negative_box=2,
        num_epochs=1,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=128,
        lora_r=8,
        lora_alpha=16,
        logging_steps=1,
        save_steps=10,
        warmup_steps=0,
        resume_from_checkpoint=None,
        format_token_weight=5.0,
        max_grad_norm=1.0,
    )


def test_stage3a_includes_negative_boxes_without_crash():
    """Reproduce and guard against the UnboundLocalError in stage3a."""
    box_samples = [{"id": f"box_{i}"} for i in range(2)]
    counting_samples = [{"id": f"count_{i}"} for i in range(2)]
    clevr_samples = [{"id": f"clevr_{i}"} for i in range(2)]
    negative_samples = [{"id": f"neg_{i}"} for i in range(2)]

    runner = MagicMock()
    runner.args = _make_args()
    runner.logger = MagicMock()

    with (
        patch.object(stage3a, "generate_coco_box_samples", return_value=box_samples),
        patch.object(stage3a, "generate_coco_counting_samples", return_value=counting_samples),
        patch.object(stage3a, "generate_clevr_spatial_dataset", return_value=clevr_samples),
        patch.object(stage3a, "generate_coco_negative_box_samples", return_value=negative_samples),
        patch.object(stage3a, "load_qlora_model", return_value=(MagicMock(), MagicMock())),
        patch.object(stage3a, "create_sft_trainer") as trainer_mock,
        patch.object(stage3a, "log_memory_status"),
        patch.object(stage3a.os.path, "exists", return_value=False),
        patch.object(stage3a.torch.cuda, "empty_cache"),
        patch("src.utils.logging_utils.setup_logging", return_value=MagicMock()),
    ):
        stage3a.train(runner)

    trainer_mock.assert_called_once()
    train_data = trainer_mock.call_args.kwargs["train_data"]
    ids = {sample["id"] for sample in train_data}

    assert all(s["id"] in ids for s in box_samples)
    assert all(s["id"] in ids for s in counting_samples)
    assert all(s["id"] in ids for s in clevr_samples)
    assert all(s["id"] in ids for s in negative_samples)

    trainer_mock.return_value.train.assert_called_once()
    trainer_mock.return_value.save_model.assert_called_once()
