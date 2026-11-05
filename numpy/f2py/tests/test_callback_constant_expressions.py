"""Constant callback expressions preserve their arithmetic and complex kinds."""
import pytest

from numpy.f2py import crackfortran
from numpy.f2py._callback_expr import literal_expression_type

from . import util


@pytest.mark.parametrize("expression,expected", [
    ("1.25_c_double + 2.5_c_double",
     {"typespec": "real", "kindselector": {"kind": "c_double"}}),
    ("1.25_8 + 2.5_4", {"typespec": "real", "kindselector": {"kind": "8"}}),
    ("1.25d0", {"typespec": "real", "kindselector": {"kind": "8"}}),
    ("(1.25d0, -2.5d0)", {"typespec": "complex", "kindselector": {"kind": "8"}}),
    ("(1.25_8, -2.5_8)", {"typespec": "complex", "kindselector": {"kind": "8"}}),
    ("(1.25_c_double, -2.5_c_double)",
     {"typespec": "complex", "kindselector": {"kind": "c_double"}}),
    ("(1_8, -2_8)", {"typespec": "complex"}),
    ("1 + 2", {"typespec": "integer"}),
    ("1.25 + 2", {"typespec": "real"}),
    ("-(1.25_c_double * 2)",
     {"typespec": "real", "kindselector": {"kind": "c_double"}}),
])
def test_constant_callback_expression_kind(expression, expected):
    assert crackfortran.determineexprtype(expression, {}) == expected


@pytest.mark.parametrize("expression", [
    "_constant_0 + 1.0_8",
    "_f2py_literal_0 + 1.0_8",
    "real(1, kind=8)",
    "1.0_kind_a + 2.0_kind_b",
])
def test_nonconstant_callback_expression_retains_caller_analysis(expression):
    assert literal_expression_type(expression) is None


@pytest.mark.slow
class TestConstantCallbackExpressions(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "constant_expressions.f90")]

    @pytest.mark.parametrize("routine,argument,result", [
        ("call_sum", 3.75, 6.25),
        ("call_double_literal", 1.25, 3.75),
        ("call_complex_literal", 1.25 - 2.5j, 3.75 - 2.5j),
    ])
    def test_native_constant_argument(self, routine, argument, result):
        observed = []

        def callback(value):
            observed.append(value)
            return value + 2.5

        function = getattr(self.module.constant_callback_expressions, routine)
        assert function(callback) == result
        assert observed == [argument]
