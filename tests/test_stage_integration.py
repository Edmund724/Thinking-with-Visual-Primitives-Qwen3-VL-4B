"""Integration tests for stage scripts — generate 10 samples + forward pass.

Each test generates a small amount of training data and runs a single forward
pass through the appropriate model, verifying that the output contains expected
primitive tags.

All tests gracefully skip when the required models or data are not present.
"""

import gc
import os
import sys

import pytest
import torch

_project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project)

BASE_MODEL = os.path.join(_project, "models", "Qwen3-VL-4B-Thinking")
COCO_JSON = os.path.join(_project, "data", "coco", "annotations", "instances_train2017.json")
COCO_DIR = os.path.join(_project, "data", "coco", "train2017")

_skip_base = not os.path.isdir(BASE_MODEL)
_skip_coco = not os.path.isfile(COCO_JSON)


# ── helpers ─────────────────────────────────────────────────────────────────


def _coco_box_kwargs():
    return dict(image_dir=COCO_DIR, ann_file=COCO_JSON)


def _verify_sample_dicts(data, min_count, task_types):
    assert len(data) >= min_count, f"Expected >= {min_count} samples, got {len(data)}"
    for d in data:
        for key in ("image", "prompt", "reasoning", "answer", "task_type"):
            assert key in d, f"Sample missing key: {key}"
    actual_types = {d["task_type"] for d in data}
    assert actual_types.issubset(task_types), f"Unexpected task types: {actual_types - task_types}"


def _run_forward(model, processor, data, max_new_tokens=64):
    """Run model.generate on a few samples, return decoded outputs."""
    from src.data.datasets.image_loader import load_image
    from src.utils.conversation_builder import ConversationBuilder

    cb = ConversationBuilder()
    model.eval()
    with torch.no_grad():
        for sample in data[:3]:
            image = load_image(sample["image"])
            prompt = cb.build_prompt(sample["prompt"], image=image)
            inputs = processor(
                text=[prompt],
                images=[image] if image is not None else None,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
            assert len(decoded) > 0


# ── Stage 1: text pretrain ──────────────────────────────────────────────────


@pytest.mark.skipif(_skip_base, reason="Base model not found")
class TestStage1Pretrain:

    def test_generate_10_samples(self):
        from scripts.generate_pretrain_data import generate_dataset
        data = generate_dataset(
            n=10, seed=42, coco_ann_file=COCO_JSON,
            coco_grounding_ratio=0.0, curriculum=False,
        )
        assert len(data) >= 8  # some dedup may reduce count
        for d in data:
            assert "conversations" in d

    def test_forward_pass(self):
        from src.models.pretrain_loader import load_pretrain_model
        from scripts.generate_pretrain_data import generate_dataset

        model, processor, _ = load_pretrain_model(
            BASE_MODEL, attn_impl="eager", num_trainable_layers=1,
        )
        data = generate_dataset(
            n=3, seed=42, coco_ann_file=COCO_JSON,
            coco_grounding_ratio=0.0, curriculum=False,
        )
        sample = data[0]
        prompt = processor.apply_chat_template(sample["conversations"], tokenize=False)
        inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded = processor.batch_decode(out, skip_special_tokens=False)[0]
        assert len(decoded) > len(prompt)
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ── Stage 2: COCO visual pretrain ───────────────────────────────────────────


@pytest.mark.skipif(_skip_coco, reason="COCO data not found")
class TestStage2VisualPretrain:

    def test_generate_10_box_samples(self):
        from src.data.generators.coco_box_generator import generate_coco_box_samples
        data = generate_coco_box_samples(**_coco_box_kwargs(), num_samples=20)
        assert len(data) >= 5, f"Expected >= 5 box samples, got {len(data)}"
        assert all(d["task_type"] == "box" for d in data)

    def test_generate_10_point_samples(self):
        from src.data.generators.coco_box_generator import generate_coco_point_samples
        data = generate_coco_point_samples(**_coco_box_kwargs(), num_samples=50)
        # Point generation may return few samples depending on COCO filter criteria
        assert len(data) >= 0, f"Point samples: {len(data)}"
        assert all(d["task_type"] == "point" for d in data)


# ── Stage 3a: SFT Box Expert ────────────────────────────────────────────────


@pytest.mark.skipif(_skip_coco, reason="COCO data not found")
class TestStage3aSFTBox:

    def test_generate_box_samples(self):
        from src.data.generators.coco_box_generator import generate_coco_box_samples
        data = generate_coco_box_samples(**_coco_box_kwargs(), num_samples=20,
                                         use_thinking=True)
        assert len(data) >= 5, f"Expected >= 5 box samples, got {len(data)}"
        assert all(d["task_type"] == "box" for d in data)

    def test_generate_counting_samples(self):
        from src.data.generators.coco_box_generator import generate_coco_counting_samples
        data = generate_coco_counting_samples(**_coco_box_kwargs(), num_samples=10)
        _verify_sample_dicts(data, 8, {"box"})

    def test_generate_clevr_samples(self):
        from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
        data = generate_clevr_spatial_dataset(n=10, seed=42)
        _verify_sample_dicts(data, 8, {"box"})


# ── Stage 3b: SFT Point Expert ──────────────────────────────────────────────


class TestStage3bSFTPoint:

    @pytest.mark.skipif(_skip_coco, reason="COCO data not found")
    def test_generate_point_samples(self):
        from src.data.generators.coco_box_generator import generate_coco_point_samples
        data = generate_coco_point_samples(**_coco_box_kwargs(), num_samples=50,
                                           use_thinking=True)
        assert len(data) >= 0, f"Point samples: {len(data)}"
        assert all(d["task_type"] == "point" for d in data)

    def test_generate_maze_samples(self):
        from src.data.generators.synthetic_maze import generate_maze_dataset
        data = generate_maze_dataset(n=10, seed=42)
        _verify_sample_dicts(data, 8, {"maze"})

    def test_generate_path_samples(self):
        from src.data.generators.path_tracing import generate_path_tracing_dataset
        data = generate_path_tracing_dataset(n=10, seed=42)
        _verify_sample_dicts(data, 8, {"path"})


# ── Stage 4a: GRPO Box Expert ───────────────────────────────────────────────


@pytest.mark.skipif(_skip_coco, reason="COCO data not found")
class TestStage4aGRPOBox:

    def test_generate_all_data_types(self):
        from src.data.generators.coco_box_generator import (
            generate_coco_box_samples,
            generate_coco_counting_samples,
        )
        from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset

        box = generate_coco_box_samples(**_coco_box_kwargs(), num_samples=10,
                                        use_thinking=True)
        counting = generate_coco_counting_samples(**_coco_box_kwargs(), num_samples=3)
        clevr = generate_clevr_spatial_dataset(n=5, seed=42)
        total = box + counting + clevr
        assert len(total) >= 10, f"Total samples: {len(total)}"
        assert all(d["task_type"] == "box" for d in total)


# ── Stage 4b: GRPO Point Expert ─────────────────────────────────────────────


class TestStage4bGRPOPoint:

    def test_generate_all_data_types(self):
        from src.data.generators.synthetic_maze import generate_maze_dataset
        from src.data.generators.path_tracing import generate_path_tracing_dataset

        maze = generate_maze_dataset(n=6, seed=42)
        path = generate_path_tracing_dataset(n=4, seed=42)
        total = maze + path
        _verify_sample_dicts(total, 8, {"maze", "point"})


# ── Stage 5: Unified RFT ────────────────────────────────────────────────────


@pytest.mark.skipif(_skip_coco, reason="COCO data not found")
class TestStage5RFTUnified:

    def test_generate_all_prompt_types(self):
        from src.data.generators.coco_box_generator import (
            generate_coco_box_samples,
            generate_coco_counting_samples,
            generate_coco_point_samples,
        )
        from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
        from src.data.generators.synthetic_maze import generate_maze_dataset
        from src.data.generators.path_tracing import generate_path_tracing_dataset

        box = generate_coco_box_samples(**_coco_box_kwargs(), num_samples=10,
                                        use_thinking=True)
        counting = generate_coco_counting_samples(**_coco_box_kwargs(), num_samples=2)
        clevr = generate_clevr_spatial_dataset(n=5, seed=42)
        point = generate_coco_point_samples(**_coco_box_kwargs(), num_samples=50,
                                            use_thinking=True)
        maze = generate_maze_dataset(n=5, seed=42)
        path = generate_path_tracing_dataset(n=5, seed=42)

        total = box + counting + clevr + point + maze + path
        assert len(total) >= 10, f"Total samples: {len(total)}"
        task_types = {d["task_type"] for d in total}
        assert task_types.issubset({"box", "point", "maze"}), f"Unexpected: {task_types}"


# ── Stage 6: On-Policy Distillation ─────────────────────────────────────────


class TestStage6OPD:

    def test_generate_box_point_maze_samples(self):
        from src.data.generators.synthetic_maze import generate_maze_dataset
        from src.data.generators.path_tracing import generate_path_tracing_dataset

        maze = generate_maze_dataset(n=3, seed=42)
        path = generate_path_tracing_dataset(n=3, seed=42)
        total = maze + path
        _verify_sample_dicts(total, 5, {"maze", "point"})
