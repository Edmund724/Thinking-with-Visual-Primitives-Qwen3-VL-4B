"""Stage orchestration — shared boilerplate for all training stage scripts.

Replaces per-script copies of:

* ``os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ...)``
* ``sys.path.insert(0, ...)``
* ``setup_logging(...)``
* ``argparse.ArgumentParser(...)`` + ``apply_yaml_defaults(...)``
* ``torch.cuda.empty_cache()`` banners
* ``pickle`` data-cache pattern

Usage (in a stage script)::

    from src.training.stage_runner import StageRunner

    runner = StageRunner("stage_name", "configs/stage_name.yaml")

    # Add stage-specific CLI args (replaces ``parser.add_argument(...)``).
    runner.add_arg("--model_path", default="models/Qwen3-VL-4B-Thinking")

    def train(runner: StageRunner) -> None:
        '''Training logic — runner owns self.args and self.logger.'''
        args, logger = runner.args, runner.logger
        model, processor = load_qlora_model(args.model_path, ...)
        data = runner.cached_data("data/cache.pkl", generate_fn)
        ...

    if __name__ == "__main__":
        runner.run(train)
"""

from __future__ import annotations

import argparse
import atexit
import glob
import logging
import os
import pickle
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from ..utils.config_utils import apply_yaml_defaults
from ..utils.logging_utils import setup_logging

# Ensure ``src/`` is importable regardless of where the script is launched.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class StageRunner:
    """Shared orchestration for a training stage.

    Parameters
    ----------
    stage_name:
        Short identifier used for the log file: ``logs/{stage_name}.log``.
    config_path:
        YAML config file path (relative to project root).  Serves as the
        default argument source; CLI flags override YAML values.
    description:
        Help text for the argparse parser.
    """

    __slots__ = ("stage_name", "config_path", "parser", "args", "logger")

    def __init__(
        self,
        stage_name: str,
        config_path: str,
        description: str = "",
    ) -> None:
        # Use a modest max-split size to reduce fragmentation from variable-length
        # sequences without enabling expandable segments, which triggers
        # "CUDA driver error: device not ready" on the 5090D/Bexus driver stack.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

        self.stage_name = stage_name
        self.config_path = config_path

        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument(
            "--config", type=str, default=None,
            help="Override the YAML config path (defaults to stage's built-in config).",
        )

        # Populated by parse_args().
        self.args: argparse.Namespace | None = None
        self.logger: logging.Logger | None = None

    # ── argument registration ─────────────────────────────────────────

    def add_arg(self, *args: Any, **kwargs: Any) -> None:
        """Register a CLI argument on the internal parser.

        A thin wrapper over ``argparse.ArgumentParser.add_argument``.
        Call this after construction but before ``parse_args()`` / ``run()``.
        """
        self.parser.add_argument(*args, **kwargs)

    def add_common_args(self) -> None:
        """Register args shared by multiple training stages.

        Currently adds the standard resume/regenerate flags so they appear
        consistently across stage scripts.
        """
        self.add_arg(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="Path to checkpoint dir to resume SFT training, e.g. outputs/stage5_rft_unified/checkpoint-500",
        )
        self.add_arg(
            "--regenerate_data",
            action="store_true",
            help="Force regeneration of prompts and filtered data, ignoring existing caches.",
        )
        self.add_arg(
            "--skip_expert_generation",
            action="store_true",
            help="Skip the expert rollout generation / difficulty grading step. "
                 "Use this when you already have prepared training data or want to "
                 "go straight to SFT training.",
        )
        self.add_arg(
            "--train_data_path",
            type=str,
            default=None,
            help="Path to a pickle file containing pre-filtered training records. "
                 "Used only when --skip_expert_generation is set.",
        )

    def parse_args(self, cli_args: list[str] | None = None) -> argparse.Namespace:
        """Parse CLI args, overlay YAML defaults, and create the logger.

        Called automatically by ``run()``, so you only need to call this
        directly when you want to inspect ``self.args`` before training.
        """
        self.args = self.parser.parse_args(cli_args)
        if self.args.config is not None:
            self.config_path = self.args.config
        apply_yaml_defaults(self.args, self.parser, self.config_path)
        self.logger = setup_logging(log_file=f"logs/{self.stage_name}.log")
        return self.args

    # ── data cache helper ─────────────────────────────────────────────

    def cached_data(
        self,
        cache_path: str,
        generate_fn: Callable[[], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Load data from *cache_path* if it exists, otherwise generate & cache.

        This replaces the repetitive ``os.path.exists + pickle.load/dump``
        pattern used in stages 3b, 4a, 4b, 5, and 6.
        """
        if os.path.exists(cache_path):
            self.logger.info(f"Loading cached data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        self.logger.info("Generating data (no cache found)...")
        data = generate_fn()
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Cached {len(data)} samples to {cache_path}")
        return data

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the latest ``checkpoint-*`` directory under *output_dir*.

        The returned path is the directory with the largest step number, or
        ``None`` if no checkpoint directory exists.  This is used by all
        training stages to resume automatically after an interruption.
        """
        if not os.path.isdir(output_dir):
            return None
        checkpoints = []
        for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
            if not os.path.isdir(path):
                continue
            m = re.search(r"checkpoint-(\d+)$", path)
            if m:
                checkpoints.append((int(m.group(1)), path))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    # ── execution ─────────────────────────────────────────────────────

    def run(self, train_fn: Callable[["StageRunner"], None]) -> None:
        """Parse args, print banner, call *train_fn*, then clean up.

        *train_fn* receives ``self`` and can access ``self.args`` and
        ``self.logger`` directly.
        """
        if self.args is None:
            self.parse_args()

        self.logger.info("=" * 60)
        self.logger.info(f"Stage: {self.stage_name}")
        self.logger.info("=" * 60)

        completed_normally = False
        interrupted = False

        def _flush_log() -> None:
            for handler in self.logger.handlers:
                handler.flush()

        def _log_unexpected_exit() -> None:
            if not completed_normally:
                self.logger.info(
                    f"Process terminated/interrupted at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
                _flush_log()

        # Log unexpected exits (SIGTERM, SIGINT via KeyboardInterrupt, crashes).
        # The transformers Trainer may install its own handlers later; atexit will
        # still run as long as the active handler eventually exits the process.
        atexit.register(_log_unexpected_exit)

        def _sigterm_handler(signum: int, frame: Any) -> None:
            nonlocal interrupted
            interrupted = True
            self.logger.info(
                f"Received SIGTERM at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; exiting..."
            )
            _flush_log()
            sys.exit(1)

        signal.signal(signal.SIGTERM, _sigterm_handler)

        torch.cuda.empty_cache()
        try:
            train_fn(self)
            completed_normally = not interrupted
        finally:
            torch.cuda.empty_cache()
            _flush_log()
            if completed_normally:
                self.logger.info(
                    f"Stage completed normally at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )

