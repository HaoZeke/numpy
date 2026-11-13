import pytest

import numpy as np
from numpy.f2py import crackfortran

from . import util


@pytest.mark.slow
class TestParameters(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [
        util.getpath("tests", "src", "parameter", "constant_real.f90"),
        util.getpath("tests", "src", "parameter", "constant_integer.f90"),
        util.getpath("tests", "src", "parameter", "constant_both.f90"),
        util.getpath("tests", "src", "parameter", "constant_compound.f90"),
        util.getpath("tests", "src", "parameter", "constant_non_compound.f90"),
        util.getpath("tests", "src", "parameter", "constant_array.f90"),
    ]

    def test_constant_real_single(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo_single, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo_single(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    def test_constant_real_double(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_double, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_double(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    def test_constant_compound_int(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int32)[::2]
        pytest.raises(ValueError, self.module.foo_compound_int, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int32)
        self.module.foo_compound_int(x)
        assert np.allclose(x, [0 + 1 + 2 * 6, 1, 2])

    def test_constant_non_compound_int(self):
        # check values
        x = np.arange(4, dtype=np.int32)
        self.module.foo_non_compound_int(x)
        assert np.allclose(x, [0 + 1 + 2 + 3 * 4, 1, 2, 3])

    def test_constant_integer_int(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int32)[::2]
        pytest.raises(ValueError, self.module.foo_int, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int32)
        self.module.foo_int(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    def test_constant_integer_long(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int64)[::2]
        pytest.raises(ValueError, self.module.foo_long, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int64)
        self.module.foo_long(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    def test_constant_both(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo(x)
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    def test_constant_no(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_no, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_no(x)
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    def test_constant_sum(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_sum, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_sum(x)
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    def test_constant_array(self):
        x = np.arange(3, dtype=np.float64)
        y = np.arange(5, dtype=np.float64)
        z = self.module.foo_array(x, y)
        assert np.allclose(x, [0.0, 1. / 10, 2. / 10])
        assert np.allclose(y, [0.0, 1. * 10, 2. * 10, 3. * 10, 4. * 10])
        assert np.allclose(z, 19.0)

    def test_constant_array_any_index(self):
        x = np.arange(6, dtype=np.float64)
        y = self.module.foo_array_any_index(x)
        assert np.allclose(y, x.reshape((2, 3), order='F'))

    def test_constant_array_delims(self):
        x = self.module.foo_array_delims()
        assert x == 9


def test_real_literal_parameter_is_value(tmp_path):
    # 1.0d0 is the number 1, so int() of it is 1. kind() still reports
    # the width, and precision() still reports 15 digits.
    src = tmp_path / "real_literal.f90"
    src.write_text(
        "subroutine s(a, b)\n"
        "  real, parameter :: x = 1.0d0\n"
        "  real, parameter :: y = 1.0e0\n"
        "  integer, parameter :: n = int(x)\n"
        "  integer, parameter :: m = int(y)\n"
        "  real a(n)\n"
        "  real b(m)\n"
        "end subroutine\n"
    )
    block = crackfortran.crackfortran([str(src)])[0]
    assert block['vars']['a']['dimension'] == ['1']
    assert block['vars']['b']['dimension'] == ['1']
    src2 = tmp_path / "real_literal_kind.f90"
    src2.write_text(
        "subroutine s(a, b)\n"
        "  integer, parameter :: dp = kind(1.0d0)\n"
        "  integer, parameter :: ep = kind(1.0e0)\n"
        "  integer, parameter :: srk = selected_real_kind(precision(1.0d0))\n"
        "  real(dp) a(srk)\n"
        "  real(ep) b(ep)\n"
        "end subroutine\n"
    )
    block = crackfortran.crackfortran([str(src2)])[0]
    assert block['vars']['a']['dimension'] == ['8']
    assert block['vars']['a']['kindselector']['kind'] == '8'
    assert block['vars']['b']['dimension'] == ['4']
    assert block['vars']['b']['kindselector']['kind'] == '4'


@pytest.mark.slow
class TestRealLiteralKind(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine from_d(a)
  real, parameter :: x = 1.0d0
  integer, parameter :: n = int(x)
  real a(n)
  a = 1.0
end subroutine

subroutine from_e(a)
  real, parameter :: x = 1.0e0
  integer, parameter :: n = int(x)
  real a(n)
  a = 1.0
end subroutine
"""

    def test_size_follows_value_not_kind(self):
        for fn in (self.module.from_d, self.module.from_e):
            one = np.ones(1, dtype=np.float32)
            fn(one)
            assert one[0] == np.float32(1.0)
            eight = np.ones(8, dtype=np.float32)
            with pytest.raises(ValueError):
                fn(eight)
