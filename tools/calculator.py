import ast
import operator

from langchain_core.tools import tool

from custom_exception.exceptions import ToolExecutionError, ToolInputError
from tools.schemas import CalculatorInput

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node: ast.expr) -> float:
    """Recursively evaluate a validated arithmetic AST."""

    match node:
        # Use exact type checking because bool is a subclass of int.
        case ast.Constant(value=value) if type(value) in (int, float):
            return float(value)

        case ast.BinOp(left=left, op=op, right=right) if type(op) in _OPS:
            return _OPS[type(op)](_eval(left), _eval(right))

        case ast.UnaryOp(op=op, operand=operand) if type(op) in _OPS:
            return _OPS[type(op)](_eval(operand))

        case _:
            raise ToolInputError(
                f"Unsupported expression node: {type(node).__name__}"
            )


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression.

    Use this tool when a task requires numeric calculations such as
    addition, subtraction, multiplication, division, or parentheses.
    Variables, functions, and arbitrary Python expressions are not
    supported.

    Args:
        expression: An arithmetic expression, e.g. "2 + 3 * 4".

    Returns:
        The numeric result of the expression.

    Raises:
        ToolInputError:
            If the expression is malformed or contains unsupported syntax.

        ToolExecutionError:
            If evaluation fails after successful validation (e.g. division
            by zero).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError(
            f"Invalid expression syntax: {expression!r}"
        ) from exc

    try:
        return _eval(tree.body)

    except ToolInputError:
        raise

    except ZeroDivisionError as exc:
        raise ToolExecutionError(
            f"Division by zero while evaluating expression: {expression!r}"
        ) from exc

    except Exception as exc:
        raise ToolExecutionError(
            f"Could not evaluate expression: {expression!r}"
        ) from exc
