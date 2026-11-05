"""Callback expressions keep their argument boundary and declared kinds."""
import pytest

from numpy.f2py import crackfortran

from . import util


@pytest.mark.parametrize("source,argument,result", [
    ("hook(1.25_c_double) result(answer)", "1.25_c_double", "answer"),
    ("hook(value + 1.25_c_double) result(answer)", "value + 1.25_c_double", "answer"),
    ("hook((value + 1.25_c_double)) result(answer)",
     "(value + 1.25_c_double)", "answer"),
    ("hook(1.25_c_double)", "1.25_c_double", None),
])
def test_callback_expression_result_boundary(source, argument, result):
    name, args, actual_result, binding = crackfortran._resolvenameargspattern(source)
    assert name == "hook"
    assert args == argument
    assert actual_result == result
    assert binding is None


@pytest.mark.parametrize("literal,typespec,kind", [
    ("1.25_c_double", "real", "c_double"),
    ("1.25_8", "real", "8"),
    ("-1.25_c_double", "real", "c_double"),
    ("1_c_long_long", "integer", "c_long_long"),
    ("1_8", "integer", "8"),
    ("-1_8", "integer", "8"),
    ("1.25", "real", None),
    ("1", "integer", None),
])
def test_callback_literal_kind(literal, typespec, kind):
    expected = {"typespec": typespec}
    if kind is not None:
        expected["kindselector"] = {"kind": kind}
    assert crackfortran.determineexprtype(literal, {}) == expected


@pytest.mark.slow
class TestCallbackExpressionResults(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "expression_results.f90")]

    @pytest.mark.parametrize("routine", [
        "call_named_kind", "call_numeric_kind", "call_expression",
    ])
    def test_native_expression_argument_and_result(self, routine):
        observed = []

        def hook(value):
            observed.append(value)
            return value + 2.5

        function = getattr(self.module.callback_expression_results, routine)
        result = function(hook, 2.0) if routine == "call_expression" else function(hook)
        expected = 3.25 if routine == "call_expression" else 1.25
        assert observed == [expected]
        assert result == expected + 2.5
