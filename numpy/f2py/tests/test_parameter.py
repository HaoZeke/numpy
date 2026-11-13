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


def test_bracket_array_constructor(tmp_path):
    # A comma inside [4, 5] is part of the constructor, and p(1) is 4.
    src = tmp_path / "brackets.f90"
    src.write_text(
        "subroutine s(a)\n"
        "  integer, parameter :: n = 2\n"
        "  integer, parameter :: p(n) = [4, 5]\n"
        "  integer, parameter :: m = p(1)\n"
        "  real a(m)\n"
        "end subroutine\n"
    )
    block = crackfortran.crackfortran([str(src)])[0]
    assert block['vars']['a']['dimension'] == ['4']


@pytest.mark.slow
class TestBracketArrayConstructor(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine s(a)
  integer, parameter :: n = 2
  integer, parameter :: p(n) = [4, 5]
  integer, parameter :: m = p(1)
  real a(m)
  a = 1.0
end subroutine
"""

    def test_bound_is_first_element(self):
        four = np.ones(4, dtype=np.float32)
        self.module.s(four)
        assert four[0] == np.float32(1.0)
        one = np.ones(1, dtype=np.float32)
        with pytest.raises(ValueError):
            self.module.s(one)
