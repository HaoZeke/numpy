import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


@pytest.mark.slow
class TestInoutByteorder(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine inc_i4(x)
  integer(kind=4), intent(inout) :: x
  x = x + 1_4
end subroutine inc_i4

subroutine inc_c8(x)
  complex(kind=8), intent(inout) :: x
  x = x + (1.0d0, 0.0d0)
end subroutine inc_c8
"""

    def test_scalar_byteorder(self):
        for dt in (">i4", "<i4"):
            x = np.array([7, 1], dtype=dt)
            self.module.inc_i4(x)
            assert_array_equal(x, np.array([8, 1], dtype=dt))
            assert x.dtype == np.dtype(dt)
        wide = np.array([7.0], dtype=">f8")
        self.module.inc_i4(wide)
        assert_array_equal(wide, np.array([8.0], dtype=">f8"))
        for dt in (">c16", "<c16"):
            z = np.array([1 + 2j], dtype=dt)
            self.module.inc_c8(z)
            assert_array_equal(z, np.array([2 + 2j], dtype=dt))
            assert z.dtype == np.dtype(dt)
