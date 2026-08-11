"""Safe, deterministic, unit-preserving arithmetic for prophetic claims."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any


class ArithmeticPolicyError(ValueError):
    """Raised when an expression violates deterministic arithmetic policy."""


@dataclass(frozen=True, slots=True)
class CalculationResult:
    expression: str
    value: Decimal
    unit: str
    assumptions: tuple[str, ...] = ()
    rule_id: str = ""

    def as_text(self) -> str:
        return f"{self.value.normalize()} {self.unit}".strip()


class SafeDecimalEvaluator(ast.NodeVisitor):
    """Evaluate a small arithmetic grammar without names, calls, or Python eval."""

    _MAX_NODES = 40
    _MAX_ABSOLUTE_VALUE = Decimal("1e12")

    def __init__(self) -> None:
        self._node_count = 0

    def evaluate(self, expression: str) -> Decimal:
        if len(expression) > 200:
            raise ArithmeticPolicyError("expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
            with localcontext() as context:
                context.prec = 28
                value = self.visit(tree)
        except (SyntaxError, DivisionByZero, InvalidOperation, OverflowError) as exc:
            raise ArithmeticPolicyError(f"invalid arithmetic expression: {expression}") from exc
        if not value.is_finite() or abs(value) > self._MAX_ABSOLUTE_VALUE:
            raise ArithmeticPolicyError("arithmetic result is outside the allowed range")
        return value

    def generic_visit(self, node: ast.AST) -> Any:
        raise ArithmeticPolicyError(f"unsupported arithmetic syntax: {type(node).__name__}")

    def visit(self, node: ast.AST) -> Decimal:
        self._node_count += 1
        if self._node_count > self._MAX_NODES:
            raise ArithmeticPolicyError("expression is too complex")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> Decimal:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Decimal:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ArithmeticPolicyError("only numeric constants are allowed")
        return Decimal(str(node.value))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Decimal:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ArithmeticPolicyError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Decimal:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ArithmeticPolicyError("division by zero")
            return left / right
        raise ArithmeticPolicyError("only +, -, *, and / are allowed")


def calculate(
    expression: str, unit: str, *, assumptions: tuple[str, ...] = (), rule_id: str = ""
) -> CalculationResult:
    if not unit.strip():
        raise ArithmeticPolicyError("result unit is required")
    return CalculationResult(
        expression=expression,
        value=SafeDecimalEvaluator().evaluate(expression),
        unit=unit,
        assumptions=assumptions,
        rule_id=rule_id,
    )


def verify_equation(equation: str) -> bool:
    if equation.count("=") != 1:
        raise ArithmeticPolicyError("equation must contain exactly one equals sign")
    left, right = (part.strip() for part in equation.split("=", maxsplit=1))
    return SafeDecimalEvaluator().evaluate(left) == SafeDecimalEvaluator().evaluate(right)


def months_to_schematic_days(
    months: Decimal | int | str, *, days_per_month: int = 30
) -> CalculationResult:
    value = Decimal(str(months))
    return calculate(
        f"{value} * {days_per_month}",
        "days",
        assumptions=(f"schematic month = {days_per_month} days",),
        rule_id="PR-001",
    )


def schematic_years_to_days(
    years: Decimal | int | str, *, days_per_year: int = 360
) -> CalculationResult:
    value = Decimal(str(years))
    return calculate(
        f"{value} * {days_per_year}",
        "days",
        assumptions=(f"schematic year = {days_per_year} days",),
        rule_id="PR-002",
    )
