import pytest

from numpy.f2py import crackfortran

from . import util


@pytest.mark.parametrize("dimspec, expression, expected", [
    ("(2)", "(/ 3, 5 /)", {1: 3, 2: 5}),
    ("(-1:1)", "(/ 6, 3, 1 /)", {-1: 6, 0: 3, 1: 1}),
])
@pytest.mark.parametrize("parameter_first", [True, False])
def test_raw_integer_parameter_array_uses_fortran_indices(
        dimspec, expression, expected, parameter_first):
    attributes = ["parameter", "dimension" + dimspec]
    if not parameter_first:
        attributes.reverse()
    variable = dict(typespec="integer", attrspec=attributes)
    variable["="] = expression
    parameters = crackfortran.get_parameters(dict(values=variable))
    assert parameters["values"] == expected
    assert isinstance(parameters["values"], dict)


def test_typed_scalar_parameter_and_raw_array_share_values():
    scalar = dict(typespec="integer", attrspec=["parameter"])
    scalar["="] = "5 / 2 * 2"
    array = dict(typespec="integer", attrspec=["parameter", "dimension(2)"])
    array["="] = "(/ n, 1 /)"
    parameters = crackfortran.get_parameters(dict(n=scalar, values=array))
    assert parameters["n"] == 4
    assert type(parameters["n"]) is int
    assert parameters["values"] == {1: 4, 2: 1}


def test_original_parameter_array_fixture_keeps_indexed_extents():
    source = util.getpath("tests", "src", "parameter", "constant_array.f90")
    blocks = {
        block["name"]: block
        for block in crackfortran.crackfortran([str(source)])
    }
    variables = blocks["foo_array"]["vars"]
    assert variables["x"]["dimension"] == ["3"]
    assert variables["y"]["dimension"] == ["5"]
    variables = blocks["foo_array_any_index"]["vars"]
    assert variables["x"]["dimension"] == ["6"]
    assert variables["y"]["dimension"] == ["2", "3"]
