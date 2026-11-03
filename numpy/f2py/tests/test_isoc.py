import pytest

import numpy as np
from numpy.testing import assert_allclose

from . import util


@pytest.mark.slow
class TestISOC(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
    def test_bindc_complex_float(self, dtype):
        a = np.array([1j, 2j, 3j], dtype=dtype)
        b = np.array([4j, 5j, 6j], dtype=dtype)
        out = {
            np.complex64: self.module.coddity.add_cfloat_arr,
            np.complex128: self.module.coddity.add_cdouble_arr,
            np.clongdouble: self.module.coddity.add_clong_double_arr
        }[dtype](a, b)
        exp = a + b
        assert_allclose(out, exp)
        assert out.dtype == exp.dtype

    # gh-24553
    def test_c_double(self):
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert out == exp_out

    # gh-9693
    def test_bindc_function(self):
        out = self.module.coddity.wat(1, 20)
        exp_out = 8
        assert out == exp_out

    # gh-25207
    def test_bindc_kinds(self):
        out = self.module.coddity.c_add_int64(1, 20)
        exp_out = 21
        assert out == exp_out

    # gh-25207 (int64/add_arr)
    @pytest.mark.parametrize("dtype", [np.int8, np.long, np.int64])
    def test_bindc_add_int_arr(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        out = {
            np.int8: self.module.coddity.add_int8_arr,
            np.long: self.module.coddity.add_clong_arr,
            np.int64: self.module.coddity.add_arr
        }[dtype](a, b)
        exp_out = a + b
        assert_allclose(out, exp_out)
        assert out.dtype == exp_out.dtype


# gh-21409
@pytest.mark.slow
class TestISOCPointerWidth(util.F2PyTest):
    sources = [util.getpath("tests", "src", "isocintrin", "gh21409.f90")]
    big = np.iinfo(np.intp).max // 2

    @pytest.mark.parametrize(
        "name", ["add_size_t_arr", "add_intptr_arr", "add_ptrdiff_arr"])
    def test_arrays(self, name):
        a = np.array([1, 2, self.big], dtype=np.intp)
        b = np.array([4, 5, 6], dtype=np.intp)
        out = getattr(self.module.ptrwidth, name)(a, b)
        assert out.dtype == np.intp
        assert out.tolist() == (a + b).tolist()

    def test_scalar(self):
        assert self.module.ptrwidth.add_size_t(self.big, 1) == self.big + 1

    def test_inout(self):
        a = np.array([1, 2, self.big], dtype=np.intp)
        self.module.ptrwidth.bump_ptrdiff(a)
        assert a.tolist() == [2, 3, self.big + 1]

    def test_function(self):
        assert self.module.ptrwidth.intptr_sum(self.big, 1) == self.big + 1


def test_only_c_long_maps_to_long():
    from numpy.f2py._isocbind import iso_c_binding_map

    # gh-21409: C long is 32 bits on Windows, so no other kind may use it
    mapped = {k for k, v in iso_c_binding_map["integer"].items() if v == "long"}
    assert mapped == {"c_long"}


def test_process_f2cmap_dict():
    from numpy.f2py.auxfuncs import process_f2cmap_dict

    f2cmap_all = {"integer": {"8": "rubbish_type"}}
    new_map = {"INTEGER": {"4": "int"}}
    c2py_map = {"int": "int", "rubbish_type": "long"}

    exp_map, exp_maptyp = ({"integer": {"8": "rubbish_type", "4": "int"}}, ["int"])

    # Call the function
    res_map, res_maptyp = process_f2cmap_dict(f2cmap_all, new_map, c2py_map)

    # Assert the result is as expected
    assert res_map == exp_map
    assert res_maptyp == exp_maptyp
