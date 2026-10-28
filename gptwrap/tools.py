"""
Tools the model can call. Each one is a normal LangChain @tool, so the
docstring is what GPT-5 reads to decide when to use it.

The calculator never calls eval(): it walks the expression's syntax tree
and only allows numbers, arithmetic and a few math functions.
"""

import ast
import math
import operator
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
    "log10": math.log10, "exp": math.exp, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        # 9**9**9 would hang the program, so keep powers reasonable
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e6 and abs(right) > 10):
            raise ValueError("that power is too big")
        return _OPS[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"'{ast.unparse(node)}' isn't allowed, only numbers, + - * / // % ** and "
                     f"{', '.join(_FUNCS)}")


def calculate(expression):
    """The calculator without the @tool wrapper, so it's easy to use and test directly."""
    expression = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expression) > 200:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"couldn't read '{expression}' as a math expression") from None
    try:
        result = _eval(tree)
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return result


@tool
def calculator(expression: str) -> str:
    """Work out a math expression exactly, e.g. "17 * 23", "sqrt(2) * 10" or "(1 + 0.07) ** 10".
    Use this for any arithmetic instead of doing it in your head."""
    try:
        return str(calculate(expression))
    except (ValueError, TypeError, OverflowError) as e:
        return f"error: {e}"


_now = datetime.now  # swapped out in the tests


@tool
def current_datetime(timezone: str = "UTC") -> str:
    """The current date, time and weekday. timezone is an IANA name like "Europe/Berlin" or "Asia/Kolkata"."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"error: unknown timezone '{timezone}', use a name like Europe/Berlin"
    now = _now(tz)
    return f"{now:%Y-%m-%d %H:%M} ({now:%A}) in {timezone}"


@tool
def days_between(start: str, end: str) -> str:
    """Number of days from start to end. Both dates as YYYY-MM-DD. Negative if end is before start."""
    try:
        d = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return "error: dates must look like 2025-03-31"
    return f"{d} days"


DEFAULT_TOOLS = [calculator, current_datetime, days_between]
