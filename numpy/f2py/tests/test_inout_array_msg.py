import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


@pytest.mark.slow
class TestInoutArrayMessage(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine bump(a, n)
  integer :: n, i
  real(8) :: a(n)
  !f2py intent(inout) :: a
  !f2py intent(hide), depend(a) :: n = len(a)
  do i = 1, n
    a(i) = a(i) + 1.0d0
  end do
end subroutine bump

subroutine bump_c(a, n)
  integer :: n, i
  real(8) :: a(n)
  !f2py intent(c,inout) :: a
  !f2py intent(hide), depend(a) :: n = len(a)
  do i = 1, n
    a(i) = a(i) + 1.0d0
  end do
end subroutine bump_c
"""

    def test_mismatch_message(self):
        native = np.array([1.0, 2.0], dtype=np.float64)
        self.module.bump(native)
        assert_array_equal(native, [2.0, 3.0])

        swapped = np.array([1.0, 2.0, 3.0], dtype=">f8")
        with pytest.raises(ValueError, match="byte order") as exc:
            self.module.bump(swapped)
        assert "not fortran contiguous" not in str(exc.value)
        assert_array_equal(swapped, np.array([1.0, 2.0, 3.0], dtype=">f8"))

        readonly = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        readonly.flags.writeable = False
        with pytest.raises(ValueError, match="writeable") as exc:
            self.module.bump(readonly)
        assert "not fortran contiguous" not in str(exc.value)
        assert_array_equal(readonly, [1.0, 2.0, 3.0])

        stepped = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)[::2]
        with pytest.raises(ValueError, match="not fortran contiguous"):
            self.module.bump(stepped)

        cview = np.arange(6, dtype=np.float64).reshape(2, 3)[:, ::2]
        with pytest.raises(ValueError, match="not contiguous") as exc:
            self.module.bump_c(cview)
        assert "fortran" not in str(exc.value)
