import pytest

import numpy as np
from numpy.testing import assert_allclose

from . import util


@pytest.mark.slow
class TestIOSF(util.F2PyTest):
    sources = [
        util.getpath('tests', 'src', 'isofenv', 'isoftests.f90'),
    ]

    def test_f_real64(self):
        out = self.module.foddity.f_add(1.0, 2.0)
        exp_out = 3
        assert out == exp_out

    def test_f_addf(self):
        out = self.module.foddity.f_addf(1.0, 2.0)
        exp_out = 3
        assert out == exp_out

    def test_wat(self):
        out = self.module.foddity.wat(1, 2)
        exp_out = 8
        assert out == exp_out

    def test_f_add_int64(self):
        out = self.module.foddity.f_add_int64(2**32, 2**32)
        exp_out = 2 ** 33
        assert out == exp_out

    def test_f_add_int16_arr(self):
        args = np.arange(6, dtype=np.int16)
        out = self.module.foddity.f_add_int16_arr(args[:3], args[3:])
        exp_out = args[:3] + args[3:]
        assert_allclose(out, exp_out)
        assert exp_out.dtype == out.dtype

    def test_f_add_int8_arr(self):
        args = np.arange(6, dtype=np.int8)
        out = self.module.foddity.f_add_int8_arr(args[:3], args[3:])
        exp_out = args[:3] + args[3:]
        assert_allclose(out, exp_out)
        assert exp_out.dtype == out.dtype

    def test_f_add_arr(self):
        args = np.arange(6, dtype=np.int64)
        out = self.module.foddity.add_arr(args[:3], args[3:])
        exp_out = args[:3] + args[3:]
        assert_allclose(out, exp_out)
        assert out.dtype == exp_out.dtype

    # gh-30352
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_f_add_complex_arr(self, dtype):
        a = np.array([1j, 2j, 3j], dtype=dtype)
        b = np.array([4j, 5j, 6j], dtype=dtype)
        out = {
            np.complex64: self.module.foddity.f_add_complex64_arr,
            np.complex128: self.module.foddity.f_add_complex128_arr,
        }[dtype](a, b)
        exp_out = a + b
        assert_allclose(out, exp_out)
        assert out.dtype == exp_out.dtype


@pytest.mark.parametrize("kind, ctype", [
    ("real32", "complex_float"),
    ("real64", "complex_double"),
])
def test_complex_real_kind_ctype(kind, ctype):
    from numpy.f2py.capi_maps import getctype

    assert getctype({"typespec": "complex", "kindselector": {"kind": kind}}) == ctype
