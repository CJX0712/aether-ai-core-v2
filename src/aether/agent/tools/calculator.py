"""计算器工具：安全求值算术表达式。"""

from __future__ import annotations

import ast
import operator
import re

from aether.core.interfaces import Tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数值")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("不支持的表达式")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


_EXPR_RE = re.compile(r"[0-9+\-*/%^().\s]+")


class CalculatorTool(Tool):
    name = "calculator"
    description = "对算术表达式求值，如 '12*8+3' 或 '(100-4)/6'。仅支持 + - * / % ^ 与括号。"
    input_param = "expression"

    def run(self, expression: str = "", **kwargs: object) -> str:
        expr = expression or str(kwargs)
        expr = expr.strip().strip("`").strip()
        matches = _EXPR_RE.findall(expr)
        target = max(matches, key=len).strip() if matches else expr.strip()
        try:
            value = safe_eval(target)
        except Exception as exc:  # noqa: BLE001
            return f"计算失败：{exc}"
        return f"{target} = {value}"
