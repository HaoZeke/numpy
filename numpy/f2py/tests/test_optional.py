import pytest

import numpy as np
from numpy.f2py import run_main

from . import util


@pytest.mark.slow
class TestOptionalPresent(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh4013.f90")]

    def test_present_branch(self):
        assert self.module.gh4013.foo(20.0) == 20.0

    def test_absent_branch(self):
        assert self.module.gh4013.foo() == 1.0

    @pytest.mark.parametrize("kwargs", [{}, {"x": None}, {"x": 0}, {"x": 20}])
    def test_scalar_presence(self, kwargs):
        expected = 1.0 if kwargs.get("x") is None else kwargs["x"]
        assert self.module.gh4013.foo(**kwargs) == expected

    def test_independent_optional_arguments(self):
        pair = self.module.gh4013.pair
        assert pair() == 0
        assert pair(0) == 100
        assert pair(None, 0) == 10
        assert pair(y=7) == 17
        assert pair(x=0, y=0) == 110

    def test_complex_presence(self):
        func = self.module.gh4013.complex_value
        assert func() == -1 - 2j
        assert func(None) == -1 - 2j
        assert func(0j) == 0j
        assert func(2 + 3j) == 2 + 3j

    def test_logical_presence(self):
        func = self.module.gh4013.logical_value
        assert func() == -1
        assert func(None) == -1
        assert func(False) == 0
        assert func(True) == 1

    def test_character_presence(self):
        func = self.module.gh4013.text_length
        assert func() == -1
        assert func(None) == -1
        assert func("") == 0
        assert func("abc") == 3

    def test_assumed_shape_presence(self):
        func = self.module.gh4013.array_size
        assert func() == -1
        assert func(None) == -1
        assert func([]) == 0
        assert func([1, 2, 3]) == 3

    def test_character_array_presence(self):
        func = self.module.gh4013.character_array_size
        assert func() == -1
        assert func(None) == -1
        assert func(np.array([], dtype="S4")) == 0
        assert func(np.array(["a", "b"], dtype="S4")) == 2

    def test_explicit_shape_presence(self):
        func = self.module.gh4013.fixed_sum
        assert func() == -1
        assert func(None) == -1
        assert func([0, 0]) == 0
        assert func([2, 3]) == 5

    def test_array_inout_presence(self):
        func = self.module.gh4013.array_inout
        assert func() == 0
        assert func(None) == 0
        values = np.array([1, 2], dtype=np.float32)
        assert func(values) == 1
        np.testing.assert_array_equal(values, [2, 4])

    @pytest.mark.parametrize("name", ["value_probe", "value_function"])
    def test_value_presence(self, name):
        func = getattr(self.module.gh4013, name)
        assert func() == -1
        assert func(None) == -1
        assert func(0) == 0
        assert func(x=23) == 23

    def test_mixed_value_arguments(self):
        func = self.module.gh4013.mixed_value
        assert func(11) == 11
        assert func(11, None) == 11
        assert func(11, 0) == 11
        assert func(11, 23) == 34

    @pytest.mark.parametrize("name", ["external_optional", "external_value"])
    def test_external_presence(self, name):
        func = getattr(self.module, name)
        assert func() == -1
        assert func(None) == -1
        assert func(0) == 0
        assert func(x=23) == 23

    def test_explicit_defaults_remain_present(self):
        for func in (self.module.gh4013.defaulted,
                     self.module.gh4013.value_defaulted,
                     self.module.external_defaulted,
                     self.module.external_value_defaulted):
            assert func() == 3
            assert func(None) == 3
            assert func(0) == 0
            assert func(23) == 23


def test_optional_wrapper_generation(tmp_path):
    source = util.getpath("tests", "src", "regression", "gh4013.f90")
    run_main(["-m", "optional_args", "--lower", "--build-dir", str(tmp_path),
              str(source)])
    wrapper = (tmp_path / "optional_args-f2pywrappers2.f90").read_text()
    normalized = "".join(wrapper.lower().split())
    assert "real,optional::x" in normalized
    value_wrapper = normalized.split(
        "subroutinef2pywrap_gh4013_value_probe(", 1)[1].split(
            "endsubroutine", 1)[0]
    assert "integer,optional::x" in value_wrapper
    assert "integer,value::base" in normalized
    assert "integer,optional,value::x" not in value_wrapper
    assert "integer,value,optional::x" not in value_wrapper
    assert "::x=3" not in normalized
    c_module = (tmp_path / "optional_argsmodule.c").read_text()
    assert "(x_capi == Py_None ? NULL : &x)" in c_module
    assert "(x_capi == Py_None ? NULL : x)" in c_module
