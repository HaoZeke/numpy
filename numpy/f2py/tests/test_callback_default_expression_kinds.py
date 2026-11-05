"""Callback constant inference follows the shared default-kind context."""
import pytest

from numpy.f2py import _callback_expr


@pytest.mark.parametrize("expression,typespec,kind", [
    ("1 + 2_4", "integer", "8"),
    ("1.0 + 2.0_4", "real", "8"),
    ("(1, 2) + (1.0_4, 2.0_4)", "complex", "8"),
    ("1.0d0", "real", "16"),
    ("(1.0d0, -2.0d0)", "complex", "16"),
])
def test_callback_expression_uses_default_kind_context(
        monkeypatch, expression, typespec, kind):
    defaults = {**_callback_expr._default_kinds,
                "integer": 8, "real": 8, "complex": 8, "double precision": 16}
    monkeypatch.setattr(_callback_expr, "_default_kinds", defaults)
    assert _callback_expr.literal_expression_type(expression) == {
        "typespec": typespec, "kindselector": {"kind": kind}}
