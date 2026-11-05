"""Native declarations retain lexical lookup and runtime integer operations."""

import ast
from pathlib import Path

import pytest

import numpy as np
from numpy.f2py import (
    _compiler_context,
    _compiler_inquiry,
    _specification_expr,
    f2py2e,
    func2subr,
)

from . import util


def _scope(variables):
    block = dict(block="subroutine", name="sample", body=[], vars=variables)
    return _specification_expr.SpecificationScopes([block], lambda block: ({}, {})).get(
        block
    )


def test_parameter_array_named_kind_is_not_an_intrinsic(monkeypatch):
    programs = []
    monkeypatch.setattr(_compiler_context, "current", lambda: object())
    monkeypatch.setattr(
        _compiler_inquiry,
        "integer_specification",
        lambda context, source: programs.append(source) or 7,
    )
    scope = _scope(
        {
            "kind": {
                "typespec": "integer",
                "attrspec": ["parameter"],
                "dimension": ["2"],
                "=": "(/7,9/)",
            },
            "chosen": {"typespec": "integer", "attrspec": ["parameter"], "=": "1"},
        }
    )
    assert scope.evaluate("kind(chosen)") == 7
    assert len(programs) == 1
    assert "dimension(2), parameter" in programs[0]
    assert "(/7,9/)" in programs[0]
    assert "kind(int(0))" not in programs[0]


def test_parameter_comparison_keeps_both_lexical_bindings(monkeypatch):
    programs = []
    monkeypatch.setattr(_compiler_context, "current", lambda: object())
    monkeypatch.setattr(
        _compiler_inquiry,
        "integer_specification",
        lambda context, source: programs.append(source) or 3,
    )
    scope = _scope(
        {
            "left": {"typespec": "integer", "attrspec": ["parameter"], "=": "2"},
            "right": {"typespec": "integer", "attrspec": ["parameter"], "=": "2"},
        }
    )
    assert scope.evaluate("merge(3,5,left==right)") == 3
    assert "left==right" not in programs[0]
    assert "= 2" in programs[0]


def test_public_only_access_entry_retains_transitive_use_binding():
    owner = dict(
        block="module",
        name="owner",
        body=[],
        vars={"value": {"typespec": "integer", "attrspec": ["parameter"], "=": "7"}},
    )
    facade = dict(
        block="module",
        name="facade",
        body=[],
        vars={"exported": {"attrspec": ["public"]}},
        use={"owner": {"only": 1, "map": {"exported": "value"}}},
    )
    consumer = dict(
        block="subroutine",
        name="consumer",
        body=[],
        vars={},
        use={"facade": {"only": 1, "map": {"local": "exported"}}},
    )
    registry = _specification_expr.SpecificationScopes(
        [owner, facade, consumer], lambda block: ({}, {})
    )
    binding = registry.get(consumer).resolve("local")
    assert binding.scope is registry.get(owner)
    assert binding.name == "value"


def test_implicit_dummy_is_a_local_binding():
    block = dict(block="subroutine", name="sample", body=[], vars={"n": {}})
    registry = _specification_expr.SpecificationScopes(
        [block], lambda block: ({"n": {"typespec": "integer"}}, {})
    )
    scope = registry.get(block)
    assert scope.resolve("n").scope is scope
    assert scope.declaration("n")["typespec"] == "integer"


@pytest.mark.parametrize(
    "expression, function",
    [
        ("n/2*2", "f2py_spec_div"),
        ("n**2", "f2py_spec_pow"),
        ("mod(a=n,p=3)", "f2py_spec_mod"),
        ("merge(n,2,n.gt.0)", "f2py_spec_merge"),
        ("abs(n)+dim(n,1)", "f2py_spec_dim"),
    ],
)
def test_runtime_operations_are_opaque_to_polynomial_simplification(
    monkeypatch,
    expression,
    function,
):
    monkeypatch.setattr(_compiler_context, "current", lambda: None)
    scope = _scope({"n": {"typespec": "integer"}})
    source = _specification_expr.runtime_integer(scope, expression)
    calls = [
        node.func.id
        for node in ast.walk(ast.parse(source, mode="eval"))
        if isinstance(node, ast.Call)
    ]
    assert function in calls
    assert not any(isinstance(node, ast.BinOp) for node in ast.walk(ast.parse(source)))


def test_native_declarations_do_not_contain_c_integer_helpers():
    variable = {
        "typespec": "integer",
        "dimension": ["f2py_spec_div(n, 2)"],
        "_fortran_dimension": {"f2py_spec_div(n, 2)": "max(0,n/2)"},
    }
    declaration = func2subr.var2fixfortran({"values": variable}, "values", f90mode=True)
    assert declaration == "integer values(max(0,n/2))"
    assert "f2py_spec_div" not in declaration


def test_native_assumed_shape_keeps_lower_bounds():
    variable = {
        "typespec": "integer",
        "dimension": ["f2py_a_d0", "f2py_a_d1"],
        "_fortran_lower_bounds": {0: "-2", 1: "5"},
    }
    assert func2subr.var2fixfortran({"a": variable}, "a", f90mode=True) == (
        "integer a(-2:(-2)+(f2py_a_d0)-1,5:(5)+(f2py_a_d1)-1)"
    )


def test_generated_array_result_separates_native_and_c_expressions(tmp_path):
    source = Path(util.getpath("specification_expr", "runtime_declarations.f90"))
    f2py2e.run_main(
        [
            "-m",
            "native_declarations",
            "--build-dir",
            str(tmp_path),
            str(source),
            "only:",
            "divided_values",
            "character_result",
            ":",
        ]
    )
    c_source = (tmp_path / "native_declarationsmodule.c").read_text()
    fortran = (tmp_path / "native_declarations-f2pywrappers2.f90").read_text()
    assert "f2py_spec_div" in c_source
    assert "f2py_spec_div" not in fortran
    assert "n/2*2" in fortran
    assert "Fortran source:" in c_source


class TestRuntimeDeclarations(util.F2PyTest):
    sources = [util.getpath("specification_expr", "runtime_declarations.f90")]

    @pytest.mark.parametrize("n", [0, 1, 5, 8])
    def test_array_result(self, n):
        result = self.module.declaration_runtime.divided_values(n)
        assert result.shape == (n // 2 * 2,)
        np.testing.assert_array_equal(result, 61)

    @pytest.mark.parametrize("n", [0, 1, 5, 8])
    def test_character_result(self, n):
        assert self.module.declaration_runtime.character_result(n) == b"x" * (
            n // 2 * 2
        )

    def test_shadowed_intrinsic(self):
        result = self.module.declaration_runtime.shadowed_inquiry()
        assert result.shape == (7,)
        np.testing.assert_array_equal(result, 67)

    @pytest.mark.parametrize("n", [0, 1, 5, 8])
    def test_integer_array_subscript(self, n):
        result = self.module.declaration_runtime.indexed_shape(
            np.array([n, 99], dtype="i")
        )
        assert result.shape == (n // 2 * 2,)
        np.testing.assert_array_equal(result, 71)

    @pytest.mark.parametrize("axis, length", [(1, 3), (2, 4)])
    def test_runtime_size(self, axis, length):
        result = self.module.declaration_runtime.selected_axis(
            np.zeros((3, 4), dtype="i"), axis
        )
        assert result.shape == (length,)
        np.testing.assert_array_equal(result, 73)

    def test_invalid_inquiry_dimension(self):
        with pytest.raises(ValueError, match="invalid DIM"):
            self.module.declaration_runtime.selected_axis(
                np.zeros((3, 4), dtype="i"), 3
            )

    @pytest.mark.parametrize("axis, length", [(1, 0), (2, 5)])
    def test_array_result_uses_actual_lower_bound(self, axis, length):
        result = self.module.declaration_runtime.bound_result(
            np.zeros((3, 4), dtype="i"), axis
        )
        assert result.shape == (length,)
        np.testing.assert_array_equal(result, 79)

    @pytest.mark.parametrize("text, length", [(b"", 0), (b"ab  ", 2), (b"a b", 3)])
    def test_runtime_character_inquiry(self, text, length):
        result = self.module.declaration_runtime.blank_shape(text)
        assert result.shape == (length,)
        np.testing.assert_array_equal(result, 83)

    @pytest.mark.parametrize("n, length", [(-5, 3), (-1, 3), (0, 0), (5, 4)])
    def test_signed_remainders(self, n, length):
        result = self.module.declaration_runtime.signed_shape(n)
        assert result.shape == (length,)
        np.testing.assert_array_equal(result, 89)

    @pytest.mark.parametrize("n, length", [(-2, 2), (0, 2), (5, 5)])
    def test_runtime_merge(self, n, length):
        result = self.module.declaration_runtime.conditional_shape(n)
        assert result.shape == (length,)
        np.testing.assert_array_equal(result, 97)

    def test_zero_divisor_raises_before_native_call(self):
        with pytest.raises(ZeroDivisionError, match="Fortran declaration"):
            self.module.declaration_runtime.division_shape(5, 0)

    @pytest.mark.parametrize("n", [0, 2, 5])
    def test_private_typed_parameter_in_array_result(self, n):
        result = self.module.declaration_runtime.private_constant_result(n)
        assert result.shape == (n + 1,)
        np.testing.assert_array_equal(result, 103)

    @pytest.mark.parametrize("n", [0, 1, 5])
    def test_rank_three_dynamic_allocation(self, n):
        assert self.module.declaration_runtime.allocate_rank_three(n) == n * 24
