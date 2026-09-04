"""
Tools the assistant can call. Each tool has a JSON-schema declaration
(passed to the LLM) and a Python implementation (executed locally when the
model requests it). Add new tools by extending TOOL_DECLARATIONS and
TOOL_IMPLEMENTATIONS with matching names.
"""
import ast
import operator
from datetime import datetime, timezone

TOOL_DECLARATIONS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression to evaluate"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current UTC date and time.",
        "parameters": {"type": "object", "properties": {}},
    },
]

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """Evaluate an arithmetic AST node without falling back to Python's eval()."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def get_current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


TOOL_IMPLEMENTATIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
}


def execute_tool(name: str, arguments: dict):
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return fn(**arguments)
