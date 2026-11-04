"""Private generic signatures retain native accessibility and C identity."""

import pytest

from numpy.f2py import _dt_helpers, _dt_namespace, _dt_operators, crackfortran

from . import util


@pytest.mark.parametrize("module_index, name", [
    (0, "default_add"), (0, "default_assign_integer"),
    (1, "explicit_add"), (1, "explicit_assign"),
])
def test_private_generic_signatures_remain_metadata(module_index, name):
    source = util.getpath("tests", "src", "derived_types", "private_operators.f90")
    module = crackfortran.crackfortran([str(source)])[module_index]
    procedure = module["f2py_operator_procedures"][name]
    assert procedure["name"] == name
    assert procedure["args"]
    assert all(argument in procedure["vars"] for argument in procedure["args"])
    assert name not in {child.get("name") for child in module["body"]}
    for variable in procedure["vars"].values():
        if variable.get("typespec") == "integer":
            assert str(variable["kindselector"]["kind"]) == "4"


def test_private_generic_does_not_gain_a_public_operator():
    source = util.getpath("tests", "src", "derived_types", "private_operators.f90")
    module = crackfortran.crackfortran([str(source)])[0]
    assert "hidden_product" in module["f2py_operator_procedures"]
    type_map, _, _ = _dt_helpers._module_types(module)
    operators = _dt_operators._scan_operator_interfaces(
        module, "private_value", type_map)
    assert "+" in operators
    assert "=" in operators
    assert "*" not in operators
    assert operators["+"][0][1]["f2py_operator_generic"] == "+"


def test_private_operator_signatures_follow_c_type_namespaces():
    source = util.getpath("tests", "src", "derived_types",
                          "private_operator_context.f90")
    modules = crackfortran.crackfortran([str(source)])
    _dt_namespace._prepare_type_namespaces(modules)
    identities = []
    for module in modules:
        native = module["f2py_operator_procedures"]["hidden_add"]
        assert native["vars"]["a"]["typename"] == "item"
        generated = _dt_namespace._c_module(module)
        type_map, order, _ = _dt_helpers._module_types(generated)
        identity = order[0]["name"]
        identities.append(identity)
        procedure = generated["f2py_operator_procedures"]["hidden_add"]
        assert procedure["vars"]["a"]["typename"] == identity
        assert procedure["vars"]["c"]["typename"] == identity
        operators = _dt_operators._scan_operator_interfaces(
            generated, identity, type_map)
        assert "+" in operators
        assert "*" not in operators
    assert len(set(identities)) == 2


@pytest.mark.slow
class TestPrivateOperatorContext(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "private_operator_context.f90")]

    @pytest.mark.parametrize("suffix, expected", [("left", 9), ("right", 109)])
    def test_public_operator_uses_its_private_module_specific(self, suffix, expected):
        module = getattr(self.module, "private_operator_owner_" + suffix)
        left, right = module.item(value=4), module.item(value=5)
        result = left + right
        assert type(result) is module.item
        assert result.value == expected
        assert left.value == 4
        assert right.value == 5
        assert not hasattr(module, "hidden_add")
        assert not hasattr(module, "hidden_product")
        with pytest.raises(TypeError):
            left * right

    @pytest.mark.parametrize("suffix, expected", [("left", 14), ("right", 21)])
    def test_public_assignment_keeps_specific_unexported(self, suffix, expected):
        module = getattr(self.module, "private_operator_owner_" + suffix)
        destination = module.item(value=3)
        assert destination.assign_from(7) is destination
        assert destination.value == expected
        assert not hasattr(module, "hidden_assign")
