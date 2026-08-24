from __future__ import annotations

import ast
import argparse
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.cli_args import parse_bool  # noqa: E402
from planning import TRACKER_MODE_CHOICES  # noqa: E402


def test_cli_options_have_one_canonical_long_name() -> None:
    for directory in (ROOT / "scripts", ROOT / "examples", ROOT / "src"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "add_argument":
                    continue
                long_options = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")
                ]
                assert len(long_options) <= 1, f"duplicate CLI aliases in {path}: {long_options}"


def test_tracker_cli_only_exposes_auto_and_pid() -> None:
    assert TRACKER_MODE_CHOICES == ("auto", "pid")


def test_cli_bool_only_accepts_true_and_false() -> None:
    assert parse_bool("true") is True
    assert parse_bool("false") is False
    for value in ("1", "0", "yes", "no", "on", "off", "TRUE", " false "):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bool(value)
