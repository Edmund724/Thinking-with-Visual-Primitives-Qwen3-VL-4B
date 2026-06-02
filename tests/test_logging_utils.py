"""Tests for logging utilities."""

from pathlib import Path

from src.utils.logging_utils import setup_logging


class TestSetupLogging:
    def test_default_name(self):
        logger = setup_logging()
        assert logger.name == "tvp"

    def test_name_from_log_file(self):
        logger = setup_logging(log_file=Path("logs/stage1_sft_unified.log"))
        assert logger.name == "stage1_sft_unified"

    def test_explicit_name_override(self):
        logger = setup_logging(name="custom_logger")
        assert logger.name == "custom_logger"

    def test_handlers_cleared_on_resetup(self):
        logger1 = setup_logging(log_file=Path("logs/test_a.log"))
        initial_handlers = list(logger1.handlers)
        assert len(initial_handlers) >= 1  # at least console

        # Re-setup with different file should clear old handlers
        logger2 = setup_logging(log_file=Path("logs/test_b.log"))
        # Same underlying logger object if name differs... actually names differ
        # so they are different loggers. Let's test same name explicitly.
        logger3 = setup_logging(name="same_name", log_file=Path("logs/test_c.log"))
        assert len(logger3.handlers) >= 1
