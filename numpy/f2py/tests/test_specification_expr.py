"""Fortran declaration arithmetic, intrinsic inquiries and dynamic allocation."""

import ast

import pytest

import numpy as np
from numpy.f2py import (
    _compiler_context,
    _compiler_inquiry,
    _specification_expr,
    crackfortran,
)

from . import util


@pytest.mark.parametrize("expression", ["tiny(1.0)", "n"])
def test_processor_inquiries_do_not_use_a_model_without_compiler(
    monkeypatch, expression
):
    monkeypatch.setattr(_compiler_context, "current", lambda: None)

    def model_is_not_a_compiler(*args, **kwargs):
        raise AssertionError("processor inquiry reached a numeric model")

    monkeypatch.setattr(_specification_expr._values, "_fortran_eval",
                        model_is_not_a_compiler)
    scope = scope_for({"n": dict(typespec="integer", attrspec=["parameter"],
                                  **{"=": "int(-log10(tiny(1.0)))"})})
    assert scope.evaluate(expression) is None


def test_synthetic_parameter_without_a_raw_declaration_keeps_public_evaluation():
    scope = scope_for({})
    variables = {"generated": dict(typespec="integer", attrspec=["parameter"],
                                    **{"=": "3"})}
    assert crackfortran.get_parameters(variables, _scope=scope)["generated"] == 3


def scope_for(variables, *, parent_variables=None):
    block = dict(block="subroutine", name="routine", vars=variables, body=[])
    if parent_variables is None:
        roots = [block]
    else:
        roots = [
            dict(block="module", name="owner", vars=parent_variables, body=[block])
        ]
    registry = _specification_expr.SpecificationScopes(roots, lambda block: ({}, {}))
    return registry.get(block)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("5/2*2", 4),
        ("-5/2*2", -4),
        ("7/3*3", 6),
        ("-7/3*3", -6),
        ("4/2", 2),
        ("(-2)**(-3)", 0),
    ],
)
def test_unselected_compiler_retains_fortran_integer_arithmetic(
    monkeypatch, expression, expected
):
    monkeypatch.setattr(_compiler_context, "current", lambda: None)
    scope = scope_for(
        {"n": dict(typespec="integer", attrspec=["parameter"], **{"=": expression})}
    )
    assert scope.evaluate("n") == expected


def test_public_numeric_param_eval_contract_is_preserved():
    assert crackfortran.param_eval("5/2*2", {}, {}) == 5.0
    assert crackfortran.param_eval("(-5)/2*2", {}, {}) == -5.0


def test_configured_processor_receives_typed_parameters_and_intrinsic(monkeypatch):
    compiler = object()
    programs = []
    monkeypatch.setattr(_compiler_context, "current", lambda: compiler)

    def query(context, program):
        assert context is compiler
        programs.append(program)
        return 7

    monkeypatch.setattr(_compiler_inquiry, "integer_specification", query)
    scope = scope_for(
        {
            "n": dict(
                typespec="integer",
                attrspec=["parameter"],
                **{"=": "merge(7,0,tiny(1.0)>0.0)"},
            )
        }
    )
    assert scope.evaluate("n") == 7
    assert len(programs) == 1
    assert "tiny(1.0)" in programs[0]
    assert "integer, parameter ::" in programs[0]
    assert "bit_size(" in programs[0]
    assert "write(*, '(I0)')" in programs[0]


def test_configured_compiler_failure_propagates_without_guessing(monkeypatch):
    monkeypatch.setattr(_compiler_context, "current", lambda: object())

    def query(context, program):
        raise _compiler_context.CompilerExecutionError(
            "configured target cannot execute"
        )

    monkeypatch.setattr(_compiler_inquiry, "integer_specification", query)
    scope = scope_for(
        {"n": dict(typespec="integer", attrspec=["parameter"], **{"=": "digits(1.0)"})}
    )
    with pytest.raises(
        _compiler_context.CompilerExecutionError, match="target cannot execute"
    ):
        scope.evaluate("n")


def test_len_inquiry_does_not_materialize_character_value(monkeypatch):
    programs = []
    monkeypatch.setattr(_compiler_context, "current", lambda: object())
    monkeypatch.setattr(
        _compiler_inquiry,
        "integer_specification",
        lambda context, program: programs.append(program) or 2147483648,
    )
    scope = scope_for(
        {
            "word": dict(
                typespec="character",
                charselector={"len": "2147483648"},
                attrspec=["parameter"],
                **{"=": "'x'"},
            )
        }
    )
    assert scope.evaluate("len(word)") == 2147483648
    assert len(programs) == 1
    assert "character" not in programs[0]
    assert "'x'" not in programs[0]


def test_inquiry_dim_uses_caller_and_shape_uses_declaration_scope(monkeypatch):
    programs = []
    monkeypatch.setattr(_compiler_context, "current", lambda: object())
    monkeypatch.setattr(
        _compiler_inquiry,
        "integer_specification",
        lambda context, program: programs.append(program) or 3,
    )
    scope = scope_for(
        {"i": dict(typespec="integer", attrspec=["parameter"], **{"=": "2"})},
        parent_variables={
            "i": dict(typespec="integer", attrspec=["parameter"], **{"=": "1"}),
            "values": dict(typespec="real", dimension=["2", "3"]),
        },
    )
    assert scope.evaluate("size(values,dim=i)") == 3
    assert len(programs) == 1
    assert "= 2" in programs[0]
    assert "= 1\n" not in programs[0]
    assert "shape_" in programs[0]


def test_intrinsic_module_renames_retain_compiler_identity(monkeypatch):
    block = dict(
        block="module",
        name="consumer",
        body=[],
        vars={},
        use={"iso_fortran_env": {"only": 1, "map": {"precision_kind": "real64"}}},
    )
    registry = _specification_expr.SpecificationScopes([block], lambda block: ({}, {}))
    programs = []
    monkeypatch.setattr(_compiler_context, "current", lambda: object())
    monkeypatch.setattr(
        _compiler_inquiry,
        "integer_specification",
        lambda context, program: programs.append(program) or 308,
    )
    assert (
        registry.get(block).evaluate("ceiling(-log10(tiny(1.0_precision_kind)))") == 308
    )
    assert "use, intrinsic :: iso_fortran_env, only:" in programs[0]
    assert "=> real64" in programs[0]
    assert "1.0_f2py_spec_" in programs[0]


@pytest.mark.parametrize(
    "expression, helper",
    [
        ("n/2*2", "f2py_spec_div"),
        ("-n/2*(-2)", "f2py_spec_div"),
        ("n**2", "f2py_spec_pow"),
        ("mod(n,3)", "f2py_spec_mod"),
        ("modulo(n,3)", "f2py_spec_modulo"),
    ],
)
def test_runtime_integer_operations_are_opaque_to_symbolic_cancellation(
    expression, helper
):
    scope = scope_for({"n": dict(typespec="integer")})
    result = _specification_expr.runtime_integer(scope, expression)
    calls = [
        node.func.id
        for node in ast.walk(ast.parse(result, mode="eval"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert helper in calls
    assert "n" in result


@pytest.mark.slow
class TestSpecificationIntrinsics(util.F2PyTest):
    sources = [util.getpath("tests", "src", "specification_expr", "intrinsics.f90")]

    def test_constant_bounds_and_signed_integer_order(self):
        output = self.module.specification_intrinsics_mod.constant_bounds()
        assert output.shape == (4,)
        assert np.all(output == 41)

    def test_intrinsics_and_parameter_array_lower_bound(self):
        output = self.module.specification_intrinsics_mod.intrinsic_bounds()
        assert output.shape == (7, 3)
        assert np.all(output == 43)

    def test_compiler_real_inquiry_defines_exact_native_extent(self):
        module = self.module.specification_intrinsics_mod
        output = module.inquiry_bounds()
        assert output.shape == (module.native_inquiry_extent(),)
        assert np.all(output == 47)

    @pytest.mark.parametrize("n, extent", [(0, 0), (1, 0), (5, 4), (8, 8)])
    def test_runtime_integer_division_retains_order(self, n, extent):
        output = self.module.specification_intrinsics_mod.runtime_bounds(n)
        assert output.shape == (extent,)
        assert np.all(output == 53)

    @pytest.mark.parametrize("n, extent", [(0, 0), (1, 0), (5, 4), (8, 8)])
    def test_runtime_negative_division_truncates_toward_zero(self, n, extent):
        output = self.module.specification_intrinsics_mod.runtime_signed(n)
        assert output.shape == (extent,)
        assert np.all(output == 59)

    def test_character_length_uses_integer_expression(self):
        assert self.module.specification_intrinsics_mod.copy_label() == b"ABCD"

    @pytest.mark.parametrize("n, total", [(0, 0), (1, 8), (5, 40)])
    def test_dynamic_allocation_with_nonunit_lower_bound(self, n, total):
        assert self.module.specification_intrinsics_mod.dynamic_allocation(n) == total
