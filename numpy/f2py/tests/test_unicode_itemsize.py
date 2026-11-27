import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


@pytest.mark.slow
class TestUnicodeItemsize(util.F2PyTest):
    suffix = ".f90"
    code = """
    subroutine c4_in(a, n, s)
      integer :: n, s, i
      character(len=4) :: a(n)
      !f2py intent(in) :: a
      !f2py intent(hide) :: n
      !f2py intent(out) :: s
      s = 0
      do i = 1, n
        if (a(i) == "ABCD") s = s + 1
        a(i) = "WXYZ"
      end do
    end subroutine c4_in

    subroutine c4_inplace(a, n, s)
      integer :: n, s, i
      character(len=4) :: a(n)
      !f2py intent(inplace) :: a
      !f2py intent(hide) :: n
      !f2py intent(out) :: s
      s = 0
      do i = 1, n
        if (a(i) == "ABCD") s = s + 1
        a(i) = "WXYZ"
      end do
    end subroutine c4_inplace
    """

    @pytest.mark.parametrize("func,writes", [("c4_in", False), ("c4_inplace", True)])
    def test_matching_itemsize(self, func, writes):
        # One UCS-4 code unit is four bytes, the same width as character(len=4).
        arr = np.array(["A", "B", "C", "D"], dtype="U1")
        before = arr.copy()
        got = getattr(self.module, func)(arr)
        assert got == 0
        if writes:
            assert_array_equal(arr, np.array(["W", "W", "W", "W"], dtype="U1"))
        else:
            assert_array_equal(arr, before)
        assert arr.dtype == before.dtype
