"""Tests for pretrain data format and embedding injection."""

import json
import tempfile
import os
import pytest


class TestPretrainDataFormat:
    """Verify generated pretrain data meets all constraints."""

    def _assistant_content(self, sample):
        """Return the assistant message content (last message)."""
        return sample["conversations"][-1]["content"]

    def test_tags_paired(self):
        """All box/point tags must be properly paired."""
        from scripts.generate_pretrain_data import generate_dataset, _validate_tags

        data = generate_dataset(n=200, seed=0)
        for sample in data:
            text = " ".join(m["content"] for m in sample["conversations"])
            assert _validate_tags(text), \
                f"Unpaired tags in: {self._assistant_content(sample)[:60]}..."

    def test_contains_primitives(self):
        """Every sample must contain at least one visual primitive tag."""
        from scripts.generate_pretrain_data import generate_dataset

        data = generate_dataset(n=200, seed=0)
        for sample in data:
            assistant = self._assistant_content(sample)
            has_box = "<|box|>" in assistant
            has_point = "<|point|>" in assistant
            assert has_box or has_point, \
                f"No primitive tag in: {assistant[:60]}..."

    def test_coordinates_in_range(self):
        """All coordinates must be integers in [0, 999]."""
        import re
        from scripts.generate_pretrain_data import generate_dataset

        data = generate_dataset(n=200, seed=0)
        pattern = re.compile(r"\[(\d+)\s*,\s*(\d+)(?:,\s*(\d+)\s*,\s*(\d+))?\]")

        for sample in data:
            assistant = self._assistant_content(sample)
            matches = pattern.findall(assistant)
            assert len(matches) > 0, f"No coordinates in: {assistant[:60]}"
            for match in matches:
                for val in match:
                    if val:  # Skip empty capture groups
                        coord = int(val)
                        assert 0 <= coord <= 999, \
                            f"Coordinate {coord} out of [0,999] in: {assistant[:60]}"

    def test_no_images(self):
        """Pretrain data must not reference image tokens."""
        from scripts.generate_pretrain_data import generate_dataset

        data = generate_dataset(n=200, seed=0)
        for sample in data:
            user = sample["conversations"][1]["content"]
            assistant = self._assistant_content(sample)
            assert "<image>" not in user.lower()
            assert "image" not in user.lower().split()[:5]  # "Show me the image" might slip
            assert "<|vision" not in user.lower()

    def test_length_under_256(self):
        """Assistant responses should be roughly under 256 tokens."""
        from scripts.generate_pretrain_data import generate_dataset

        data = generate_dataset(n=200, seed=0)
        for sample in data:
            assistant = self._assistant_content(sample)
            # Rough heuristic: ~4 chars per token for English
            assert len(assistant.split()) <= 100, \
                f"Assistant too long ({len(assistant.split())} words): {assistant[:60]}..."

    def test_output_file_format(self):
        """Generated JSON file must be a valid list of conversations."""
        from scripts.generate_pretrain_data import generate_dataset, export_for_training

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_pretrain.json")
            data = generate_dataset(n=50, seed=0)
            export_for_training(data, path)

            with open(path) as f:
                loaded = json.load(f)

            assert isinstance(loaded, list)
            assert len(loaded) == 50
            for item in loaded:
                assert "conversations" in item
                assert len(item["conversations"]) == 3
                assert item["conversations"][0]["role"] == "system"
                assert item["conversations"][1]["role"] == "user"
                assert item["conversations"][2]["role"] == "assistant"


class TestEmbeddingInjection:
    """Verify pretrained embedding injection logic."""

    @pytest.mark.skipif(True, reason="Requires actual model loading; run manually with GPU")
    def test_injection_preserves_old_tokens(self):
        """Old vocab rows must be untouched after injection."""
        pass  # Manual test only
