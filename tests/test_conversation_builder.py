"""Tests for ConversationBuilder — message construction for all modes."""

import pytest

from src.utils.conversation_builder import ConversationBuilder


# ── helpers ────────────────────────────────────────────────────────────────

class FakeImage:
    """Minimal image stub for tests that require an image argument."""


# ── mode selection ───────────────────────────────────────────────────────────

class TestModeSelection:
    def test_sft_mode_uses_sft_system_message(self):
        cb = ConversationBuilder("sft")
        assert "visual primitives when needed" in cb.system_message
        assert "do not use characters from other languages" in cb.system_message
        assert " thinking" not in cb.system_message  # SFT doesn't instruct tags

    def test_grpo_mode_uses_grpo_system_message(self):
        cb = ConversationBuilder("grpo")
        assert "inside  thinking" in cb.system_message
        assert "Always close your reasoning with  response" in cb.system_message

    def test_opd_mode_uses_short_system_message(self):
        cb = ConversationBuilder("opd")
        assert cb.system_message == (
            "You are a helpful visual reasoning assistant. Think step by step."
        )

    def test_pretrain_mode_excludes_other_languages_clause(self):
        cb = ConversationBuilder("pretrain")
        assert "do not use characters" not in cb.system_message
        assert "Respond in English only." in cb.system_message

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            ConversationBuilder("invalid")

    def test_custom_system_message_overrides_mode(self):
        cb = ConversationBuilder("sft", system_message="Custom message")
        assert cb.system_message == "Custom message"


# ── build_user_content ─────────────────────────────────────────────────────

class TestBuildUserContent:
    def test_text_only(self):
        result = ConversationBuilder._build_user_content("hello")
        assert result == "hello"

    def test_with_image(self):
        img = FakeImage()
        result = ConversationBuilder._build_user_content("hello", img)
        assert len(result) == 2
        assert result[0] == {"type": "image", "image": img}
        assert result[1] == {"type": "text", "text": "hello"}


# ── build_prompt ────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_text_only(self):
        cb = ConversationBuilder("grpo")
        msgs = cb.build_prompt("Where is the cat?")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": cb.system_message}
        assert msgs[1] == {"role": "user", "content": "Where is the cat?"}

    def test_with_image(self):
        cb = ConversationBuilder("grpo")
        img = FakeImage()
        msgs = cb.build_prompt("Where is the cat?", img)
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": cb.system_message}
        user_content = msgs[1]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"type": "image", "image": img}
        assert user_content[1] == {"type": "text", "text": "Where is the cat?"}


# ── build_sft ───────────────────────────────────────────────────────────────

class TestBuildSFT:
    def test_basic(self):
        cb = ConversationBuilder("sft")
        sample = {
            "prompt": "Where is the cat?",
            "reasoning": "The cat is on the mat.",
            "answer": "on the mat",
        }
        prompt_msgs, full_msgs = cb.build_sft(sample)
        assert len(prompt_msgs) == 2  # system + user
        assert len(full_msgs) == 3     # system + user + assistant
        assert full_msgs[2]["role"] == "assistant"
        assert full_msgs[2]["content"] == "on the mat"
        assert full_msgs[2]["reasoning_content"] == "The cat is on the mat."

    def test_strips_think_tags_from_reasoning(self):
        cb = ConversationBuilder("sft")
        sample = {
            "prompt": "Where?",
            "reasoning": " thinking\nThe cat is here\n response",
            "answer": "here",
        }
        _, full_msgs = cb.build_sft(sample)
        assert full_msgs[2]["reasoning_content"] == "The cat is here"

    def test_leading_think_only(self):
        cb = ConversationBuilder("sft")
        sample = {
            "prompt": "Where?",
            "reasoning": " thinkingstep 1\nstep 2",
            "answer": "done",
        }
        _, full_msgs = cb.build_sft(sample)
        assert full_msgs[2]["reasoning_content"] == "step 1\nstep 2"

    def test_trailing_think_only(self):
        cb = ConversationBuilder("sft")
        sample = {
            "prompt": "Where?",
            "reasoning": "step 1\nstep 2 response",
            "answer": "done",
        }
        _, full_msgs = cb.build_sft(sample)
        assert full_msgs[2]["reasoning_content"] == "step 1\nstep 2"

    def test_empty_reasoning(self):
        cb = ConversationBuilder("sft")
        sample = {
            "prompt": "Where?",
            "reasoning": "",
            "answer": "nowhere",
        }
        _, full_msgs = cb.build_sft(sample)
        assert full_msgs[2]["reasoning_content"] == ""

    def test_prompts_are_prefix_of_full(self):
        """prompt_messages should be a strict prefix of full_messages."""
        cb = ConversationBuilder("sft")
        sample = {
            "prompt": "Where?",
            "reasoning": "thinking...",
            "answer": "answer",
        }
        prompt_msgs, full_msgs = cb.build_sft(sample)
        assert len(prompt_msgs) == 2
        assert len(full_msgs) == 3
        assert full_msgs[:2] == prompt_msgs

    def test_with_image(self):
        cb = ConversationBuilder("sft")
        img = FakeImage()
        sample = {
            "prompt": "Where?",
            "reasoning": "found it",
            "answer": "here",
            "image": img,
        }
        prompt_msgs, full_msgs = cb.build_sft(sample)
        assert len(prompt_msgs) == 2
        assert prompt_msgs[1]["content"][0] == {"type": "image", "image": img}


# ── build_pretrain ──────────────────────────────────────────────────────────

class TestBuildPretrain:
    def test_basic(self):
        cb = ConversationBuilder("pretrain")
        msgs = cb.build_pretrain("Where is the cat?", "The cat is on the mat.")
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Where is the cat?"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "The cat is on the mat."


# ── build_gt_text ──────────────────────────────────────────────────────────

class TestBuildGTText:
    def test_basic(self):
        gt = ConversationBuilder.build_gt_text("step 1\nstep 2", "42")
        expected = "step 1\nstep 2\n response\n\nThe answer is 42."
        assert gt == expected

    def test_empty_reasoning(self):
        gt = ConversationBuilder.build_gt_text("", "yes")
        assert gt == "\n response\n\nThe answer is yes."

    def test_no_answer(self):
        gt = ConversationBuilder.build_gt_text("reasoning", "")
        assert gt == "reasoning\n response\n\nThe answer is ."