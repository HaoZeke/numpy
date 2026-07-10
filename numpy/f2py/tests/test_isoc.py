import pytest

import numpy as np
from numpy.testing import assert_allclose

from . import util


@pytest.mark.slow
class TestISOC(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

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

    # gh-25207
    def test_bindc_add_arr(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        out = self.module.coddity.add_arr(a, b)
        exp_out = a * 2
        assert_allclose(out, exp_out)


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


@pytest.mark.slow
class TestISOCPointerWidths(util.F2PyTest):
    # gh-25229: c_size_t/c_intptr_t map to pointer-width npy types, not
    # to 4-byte 'unsigned'/'long' aliases that truncate on LP64/LLP64
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

    def test_c_size_t(self):
        big = 2**33 + 7
        assert self.module.coddity.c_add_sizes(big, 1) == big + 1

    def test_c_size_t_rejects_negative(self):
        with pytest.raises(OverflowError):
            self.module.coddity.c_add_sizes(-1, 1)

    def test_c_intptr_t(self):
        big = 2**33 + 7
        assert self.module.coddity.c_diff_ptrs(big, 7) == 2**33
