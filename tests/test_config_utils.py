"""Tests for YAML config utilities."""
import argparse
import tempfile
from pathlib import Path

import yaml

from src.utils.config_utils import apply_yaml_defaults, load_yaml_config


def _make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    return p


def _write_yaml(content: dict) -> Path:
    """Write a temp YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(content, tmp)
    tmp.close()
    return Path(tmp.name)


# ── load_yaml_config ─────────────────────────────────────────────────────


def test_load_missing_yaml():
    assert load_yaml_config("/nonexistent/path.yaml") == {}


def test_load_empty_yaml():
    path = _write_yaml({})
    try:
        assert load_yaml_config(path) == {}
    finally:
        path.unlink()


def test_load_yaml_with_keys():
    path = _write_yaml({"batch_size": 4, "num_epochs": 3})
    try:
        cfg = load_yaml_config(path)
        assert cfg == {"batch_size": 4, "num_epochs": 3}
    finally:
        path.unlink()


# ── apply_yaml_defaults ──────────────────────────────────────────────────


def test_yaml_key_matches_arg():
    """YAML key that matches an argparse name should override the default."""
    path = _write_yaml({"batch_size": 8, "num_epochs": 2})
    try:
        parser = _make_parser()
        args = parser.parse_args([])
        apply_yaml_defaults(args, parser, path)
        assert args.batch_size == 8
        assert args.num_epochs == 2
        assert args.learning_rate is None  # Not in YAML, stays default
    finally:
        path.unlink()


def test_yaml_key_does_not_match_arg():
    """YAML key with no matching argparse arg should be silently skipped."""
    path = _write_yaml({"per_device_batch_size": 4, "num_train_epochs": 5})
    try:
        parser = _make_parser()
        args = parser.parse_args([])
        apply_yaml_defaults(args, parser, path)
        # Neither key matches an argparse arg → both remain default None
        assert args.batch_size is None
        assert args.num_epochs is None
    finally:
        path.unlink()


def test_cli_overrides_yaml():
    """CLI argument should take precedence over YAML value."""
    path = _write_yaml({"batch_size": 8})
    try:
        parser = _make_parser()
        args = parser.parse_args(["--batch_size", "16"])
        apply_yaml_defaults(args, parser, path)
        assert args.batch_size == 16  # CLI wins
    finally:
        path.unlink()


def test_mixed_yaml_keys():
    """Some keys match, some don't — matched ones applied, others skipped."""
    path = _write_yaml({
        "batch_size": 12,
        "per_device_batch_size": 32,   # No matching arg → skipped
        "num_epochs": 5,
    })
    try:
        parser = _make_parser()
        args = parser.parse_args([])
        apply_yaml_defaults(args, parser, path)
        assert args.batch_size == 12   # Applied
        assert args.num_epochs == 5    # Applied
    finally:
        path.unlink()


def test_default_not_none():
    """When argparse default is 0 (not None), YAML should still apply."""
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=0)
    path = _write_yaml({"batch_size": 4})
    try:
        args = p.parse_args([])
        apply_yaml_defaults(args, p, path)
        assert args.batch_size == 4
    finally:
        path.unlink()
