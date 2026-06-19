"""Tests for LLM-as-Judge Quality RM (quality_rm_api.py)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.utils.quality_rm_api import (
    _build_judge_prompt,
    _load_api_config,
    _parse_score,
    make_quality_reward_api_fn,
    quality_reward_api,
)


# ── _parse_score ────────────────────────────────────────────────────────────

class TestParseScore:
    def test_score_colon_format_1_0(self):
        assert _parse_score("no issues found\nScore: 1.0") == 1.0

    def test_score_colon_format_0_5(self):
        assert _parse_score("minor redundancy\nScore: 0.5") == 0.5

    def test_score_colon_format_0_0(self):
        assert _parse_score("serious contradiction\nScore: 0.0") == 0.0

    def test_score_colon_case_insensitive(self):
        assert _parse_score("score: 1.0") == 1.0

    def test_score_colon_with_extra_text_after(self):
        assert _parse_score("Score: 0.5 (minor issues)") == 0.5

    def test_legacy_bare_1_0(self):
        assert _parse_score("1.0") == 1.0

    def test_legacy_bare_1(self):
        assert _parse_score("1") == 1.0

    def test_legacy_bare_0_5(self):
        assert _parse_score("0.5") == 0.5

    def test_legacy_bare_dot5(self):
        assert _parse_score(".5") == 0.5

    def test_legacy_bare_0_0(self):
        assert _parse_score("0.0") == 0.0

    def test_legacy_bare_0(self):
        assert _parse_score("0") == 0.0

    def test_extract_from_text(self):
        assert _parse_score("I think the answer is 1.0 for sure") == 1.0

    def test_raises_on_invalid(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_score("gibberish")


# ── _build_judge_prompt ─────────────────────────────────────────────────────

class TestBuildJudgePrompt:
    def test_contains_checklist(self):
        prompt = _build_judge_prompt("pred", "gt", "box")
        assert "Redundancy" in prompt
        assert "Consistency" in prompt
        assert "Contradiction" in prompt
        assert "Reward hacking" in prompt
        assert "Self-contradiction" in prompt
        assert "Meaningful references" in prompt

    def test_contains_input_texts(self):
        prompt = _build_judge_prompt("pred text", "gt text", "point")
        assert "pred text" in prompt
        assert "gt text" in prompt
        assert "point" in prompt

    def test_asks_for_score_line(self):
        prompt = _build_judge_prompt("p", "g", "box")
        assert "Score:" in prompt


# ── _load_api_config ────────────────────────────────────────────────────────

class TestLoadAPIConfig:
    def test_returns_empty_without_key(self):
        # dotenv may have already loaded the real .env file — force API key to None.
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            cfg = _load_api_config()
            assert cfg == {}

    def test_returns_config_with_key(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "QUALITY_RM_MODEL": "test-model",
        }, clear=True):
            cfg = _load_api_config()
            assert cfg["api_key"] == "sk-test"
            assert cfg["model"] == "test-model"

    def test_sample_ratio_default(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            cfg = _load_api_config()
            assert cfg["sample_ratio"] == 1.0

    def test_sample_ratio_custom(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "QUALITY_RM_SAMPLE_RATIO": "0.3",
        }, clear=True):
            cfg = _load_api_config()
            assert cfg["sample_ratio"] == 0.3


# ── quality_reward_api ──────────────────────────────────────────────────────

class TestQualityRewardAPI:
    def test_falls_back_when_no_config(self):
        with patch("src.utils.quality_rm_api._load_api_config", return_value={}):
            score = quality_reward_api("pred", "gt", "box")
            assert score in (0.0, 0.5, 1.0)


# ── subset sampling ─────────────────────────────────────────────────────────

class TestSubsetSampling:
    def test_all_api_when_ratio_1_0(self):
        """With sample_ratio=1.0, every completion should hit the API."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "QUALITY_RM_SAMPLE_RATIO": "1.0",
        }, clear=True):
            fn = make_quality_reward_api_fn(task_type_default="box")
            # Mock the API call so we can check it was made
            with patch("src.utils.quality_rm_api.quality_reward_api",
                       return_value=1.0) as mock_api:
                # Fake completions: list of token-id tensors
                completions = [MagicMock() for _ in range(20)]
                # Each input needs gt_text
                inputs = [{"gt_text": f"gt_{i}", "task_type": "box"} for i in range(20)]
                fn(completions, inputs=inputs)
                # All 20 should have gone through the API
                assert mock_api.call_count == 20

    def test_fallback_when_ratio_0_0(self):
        """With sample_ratio=0.0, rule-based fallback for all."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "QUALITY_RM_SAMPLE_RATIO": "0.0",
        }, clear=True):
            fn = make_quality_reward_api_fn(task_type_default="box")
            with patch("src.utils.quality_rm_api.quality_reward_api",
                       return_value=1.0) as mock_api:
                completions = [MagicMock() for _ in range(10)]
                inputs = [{"gt_text": f"gt_{i}", "task_type": "box"} for i in range(10)]
                fn(completions, inputs=inputs)
                # None should have gone through the API
                assert mock_api.call_count == 0
