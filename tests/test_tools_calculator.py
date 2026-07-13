"""Test 4 of 6+: tool schemas — covers the 'tool schemas' test category."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_exception.exceptions import ToolExecutionError, ToolInputError
from tools.calculator import calculator  # noqa: E402


def test_calculator_returns_correct_result():
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == 14


def test_calculator_rejects_unsafe_input():
    import pytest

    with pytest.raises(ToolInputError):
        calculator.invoke({"expression": "__import__('os')"})