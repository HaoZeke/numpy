"""Derived expression support preserves the numeric parameter interface."""

import textwrap

import pytest

from numpy.f2py import crackfortran
from numpy.f2py._dt_helpers import _eval_kind_expr

from . import util


@pytest.mark.parametrize("expression, expected", [
    ("kind(0)", 4),
    ("kind(1_8 * 1.0)", 4),
    ("kind((1.0d0 + 2) * 2)", 8),
    ("kind(max(1, 2_8))", 8),
    ("kind((1.0d0, 2.0d0))", 8),
])
def test_derived_kind_expression_keeps_expression_types(expression, expected):
    assert _eval_kind_expr(expression) == expected


@pytest.mark.parametrize("value", ["0.1", "3.14", "1.0 / 3"])
def test_numeric_scalar_parameter_remains_builtin_float(value):
    actual = crackfortran.param_eval(value, {}, {})
    assert type(actual) is float
    assert actual == eval(value)


def test_numeric_array_parameter_keeps_existing_float_values():
    actual = crackfortran.param_eval("(/3.14, 4./)", {}, {},
                                    dimspec="(2)")
    assert actual == {1: 3.14, 2: 4.0}
    assert type(actual[1]) is float


@pytest.mark.parametrize("kind", [4, 8, 16])
def test_named_kind_result_is_plain_integer(kind):
    parameter = {"typespec": "integer", "attrspec": ["parameter"]}
    actual = crackfortran.get_parameters({
        "requested": {**parameter, "=": str(kind)},
        "resolved": {**parameter, "=": "kind(1.0_requested)"},
    })
    assert actual["resolved"] == kind
    assert type(actual["resolved"]) is int


def test_intrinsic_parameters_respect_only_rename_and_scope(monkeypatch):
    monkeypatch.setattr(crackfortran, "f90modulevars", {})
    used = {"name": "selected", "use": {
        "iso_fortran_env": {"only": 1, "map": {"rk": "real64"}}}}
    assert crackfortran.get_useparameters(used) == {"rk": 8}
    assert crackfortran.get_useparameters({"name": "other", "use": {}}) == {}
    assert crackfortran.f90modulevars == {}


def test_empty_only_list_imports_no_intrinsic_parameters(monkeypatch):
    monkeypatch.setattr(crackfortran, "f90modulevars", {})
    used = {"name": "empty", "use": {
        "iso_fortran_env": {"only": 1, "map": {}}}}
    assert crackfortran.get_useparameters(used) == {}


def test_type_component_inquiries_and_bounds_use_host_constants(tmp_path):
    path = tmp_path / "member_inquiries.f90"
    path.write_text(textwrap.dedent("""\
        module member_inquiries
          integer, parameter :: count = 3
          type :: sample
            real :: values(-1:count-2)
            real :: pair(size([1, 2]))
            character(len=len('word')) :: name
          end type sample
        end module member_inquiries
        """))
    module = crackfortran.crackfortran([str(path)])[0]
    fields = module["body"][0]["vars"]
    assert fields["values"]["dimension"] == ["3"]
    assert fields["pair"]["dimension"] == ["2"]
    assert fields["name"]["charselector"]["len"] == "4"


@pytest.mark.slow
class TestDerivedModuleVariableBoundary(util.F2PyTest):
    suffix = ".f90"
    code = """
        module derived_variable_boundary
          use iso_c_binding, only: c_int
          type, bind(c) :: cell
            integer(c_int) :: value = 0
          end type cell
          type(cell) :: unexposed
          integer :: counter = 5
        end module derived_variable_boundary
        """

    def test_classes_and_scalar_variables_survive_unmapped_module_value(self):
        module = self.module.derived_variable_boundary
        assert not hasattr(module, "unexposed")
        assert module.counter == 5
        module.counter = 7
        assert module.counter == 7
        value = self.module.cell()
        assert value.value == 0
        value.value = 11
        assert value.value == 11
