"""Represent scalar and runtime-shaped record components without reading them."""

import pytest

import numpy as np
from numpy.f2py import _dt_codegen

from . import util


def test_scalar_nested_record_repr_uses_scalar_placeholder():
    code = _dt_codegen._gen_opaque_tp_repr("holder", {
        "scalar": {"typespec": "type", "typename": "atom"},
    })
    assert "scalar=<atom>" in code
    assert "scalar=<array>" not in code


@pytest.mark.parametrize("dimensions, count", [
    (["2", "3"], 6), (["0", "2"], 0),
], ids=["fixed", "zero-extent"])
def test_fixed_record_array_repr_retains_count(dimensions, count):
    code = _dt_codegen._gen_opaque_tp_repr("holder", {
        "fixed": {"typespec": "type", "typename": "atom",
                  "dimension": dimensions},
    })
    assert f"fixed=[{count}xatom]" in code
    assert "fixed=<array>" not in code


@pytest.mark.parametrize("dimensions, attributes", [
    ([":", ":"], ["allocatable"]),
    ([":", ":"], ["pointer"]),
    (["n", "2"], []),
], ids=["allocatable", "pointer", "len-parameter"])
def test_runtime_record_array_repr_uses_array_placeholder(dimensions, attributes):
    code = _dt_codegen._gen_opaque_tp_repr("holder", {
        "runtime": {"typespec": "type", "typename": "atom",
                    "dimension": dimensions, "attrspec": attributes},
    }, len_info=[{"name": "n"}])
    assert "runtime=<array>" in code
    assert "runtime=<atom>" not in code
    assert "f2py_get_holder_runtime(" not in code


@pytest.mark.slow
class TestRuntimeRecordRepr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types", "record_repr.f90")]

    def test_allocated_record_repr_preserves_values(self):
        value = self.module.repr_container()
        assert value.allocated is None
        value.scalar = self.module.repr_atom(x=7)
        for shape in [(2, 3), (0, 3), (1, 2)]:
            records = np.empty(shape, dtype=object)
            for index in np.ndindex(shape):
                records[index] = self.module.repr_atom(x=10 * index[0] + index[1])
            value.allocated = records
            description = repr(value)
            assert "scalar=<repr_atom>" in description
            assert "fixed=[6xrepr_atom]" in description
            assert "allocated=<array>" in description
            assert value.scalar.x == 7
            copied = value.allocated
            assert copied.shape == shape
            for index in np.ndindex(shape):
                assert copied[index].x == records[index].x
        value.allocated = None
        assert value.allocated is None
        assert "allocated=<array>" in repr(value)

    def test_pointer_record_repr_preserves_association(self):
        value = self.module.repr_container()
        assert value.section is None
        assert "section=<array>" in repr(value)
        self.module.repr_records.attach_records(value)
        description = repr(value)
        assert "section=<array>" in description
        copied = value.section
        assert copied.shape == (2, 2)
        assert copied[0, 0].x == 12
        assert copied[1, 1].x == 34
        self.module.repr_records.detach_records(value)
        assert value.section is None
        assert "section=<array>" in repr(value)
        assert copied[1, 1].x == 34
