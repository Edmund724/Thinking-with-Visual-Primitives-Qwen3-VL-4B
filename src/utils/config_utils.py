"""YAML config helpers for stage scripts.

Stage scripts support YAML-based default parameters so that `configs/*.yaml`
becomes the single source of truth for hyperparameters. Command-line arguments
still override YAML values.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Returns an empty dict if missing or empty."""
    path = Path(yaml_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def apply_yaml_defaults(args, parser, yaml_path: str | Path) -> None:
    """Apply YAML defaults to args for any argument not explicitly overridden.

    After ``parser.parse_args()`` has produced ``args``, this function loads the
    YAML config and sets ``args.<key> = value`` for every YAML key that:
      - corresponds to a registered argument, and
      - was left at its parser default (i.e., not supplied on the command line).

    This lets ``configs/*.yaml`` act as the default parameter source while
    preserving the ability to override any value from the CLI.
    """
    cfg = load_yaml_config(yaml_path)
    for key, value in cfg.items():
        if key not in args:
            # YAML contains a key that is not an argument; skip silently.
            continue
        default = parser.get_default(key)
        current = getattr(args, key)
        # Apply YAML value when the argument was not explicitly set.
        # argparse leaves overridden values as the provided type; comparing to
        # the registered default is a pragmatic proxy.
        if current == default:
            setattr(args, key, value)
