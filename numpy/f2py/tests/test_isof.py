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


@pytest.mark.slow
class TestGH23633(util.F2PyTest):
    sources = [
        util.getpath('tests', 'src', 'isofenv', 'gh23633.f90'),
    ]

    @pytest.mark.parametrize('name', ['add_renamed', 'add_intrinsic'])
    def test_real64_kind(self, name):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        fn = getattr(self.module, name)
        c = fn(a, b)
        assert c.dtype == np.dtype(np.float64)
        assert_allclose(c, [4.0, 6.0])
