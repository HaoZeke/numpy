import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

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


@pytest.mark.slow
class TestRenamedKindsSignature(util.F2PyTest):
    # gh-17563: the signature file of the issue, with renamed kinds split
    # over two USE statements
    module_name = "gh17563"
    sources = [
        util.getpath('tests', 'src', 'isofenv', 'gh17563.pyf'),
        util.getpath('tests', 'src', 'isofenv', 'gh17563.f90'),
    ]

    def test_renamed_real64(self):
        assert self.module.func(1) == 1 / 3


@pytest.mark.slow
class TestRenamedKindsModule(util.F2PyTest):
    # gh-17563: renamed kinds used by module procedures through host
    # association, including the generated Fortran wrappers
    sources = [
        util.getpath('tests', 'src', 'isofenv', 'gh17563_mod.f90'),
    ]
    # exact in double precision, rounded away in single precision
    eps = 2.0 ** -30

    def test_assumed_shape(self):
        v = np.array([1.0, 2.0, 3.0])
        n = self.module.gh17563_mod.rescale(v, 1 + self.eps)
        assert_array_equal(v, np.array([1.0, 2.0, 3.0]) * (1 + self.eps))
        assert n == 3

    def test_function_result(self):
        v = np.array([1 + self.eps, 2.0, 3.0])
        assert self.module.gh17563_mod.total(v) == 6 + self.eps

    # gh-30352
    def test_complex(self):
        arr = np.zeros(4, dtype=np.complex128)
        self.module.gh17563_mod.fill(arr)
        assert_array_equal(arr, np.full(4, 1 - 1j))

    def test_iso_c_binding(self):
        z = np.array([1 + 1j, 2 - 1j])
        self.module.gh17563_mod.cscale(z, 1 + self.eps)
        assert_array_equal(z, np.array([1 + 1j, 2 - 1j]) * (1 + self.eps))


@pytest.mark.slow
class TestKindCaseSignature(util.F2PyTest):
    # gh-17563: signature files keep their case, kind names ignore it
    module_name = "gh17563_case"
    sources = [
        util.getpath('tests', 'src', 'isofenv', 'gh17563_case.pyf'),
        util.getpath('tests', 'src', 'isofenv', 'gh17563_case.f90'),
    ]

    def test_uppercase_real64(self):
        eps = 2.0 ** -30
        assert self.module.twice(1 + eps) == 2 + 2 * eps


@pytest.mark.parametrize("kind, ctype", [
    ("real32", "complex_float"),
    ("real64", "complex_double"),
])
def test_complex_real_kind_ctype(kind, ctype):
    from numpy.f2py.capi_maps import getctype

    assert getctype({"typespec": "complex", "kindselector": {"kind": kind}}) == ctype
