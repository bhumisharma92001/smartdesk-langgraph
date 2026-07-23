"""Safe arithmetic tool."""
import ast
import operator
from langchain_core.tools import ToolException, tool
from tools.schemas import CalculatorInput

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def evaluate(node: ast.expr) -> float:
    """Evaluate numbers and allow-listed arithmetic operators only."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.operand))
    raise ValueError("unsupported expression")


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> float:
    """Safely evaluate a mathematical expression without executing Python code."""
    try:
        return evaluate(ast.parse(expression, mode="eval").body)
    except Exception as exc:
        raise ToolException(f"Calculation failed: {exc}") from exc
