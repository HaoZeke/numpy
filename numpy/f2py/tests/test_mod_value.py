import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestModValue(util.F2PyTest):
    suffix = ".f90"
    code = """
module valmod
  use iso_c_binding, only: c_int, c_double
  implicit none
contains
  subroutine addi(n, x)
    integer(c_int), value :: n
    integer(c_int), intent(inout) :: x(:)
    x = x + n
  end subroutine addi

  subroutine addd(n, x)
    real(c_double), value :: n
    real(c_double), intent(inout) :: x(:)
    x = x + n
  end subroutine addd

  function sumi(n, x) result(s)
    integer(c_int), value :: n
    integer(c_int), intent(in) :: x(:)
    integer(c_int) :: s
    s = n + sum(x)
  end function sumi
end module valmod
"""

    def test_c_int_value(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        self.module.valmod.addi(10, x)
        assert list(x) == [11, 12, 13]

    def test_c_double_value(self):
        x = np.array([1.0, 2.0], dtype=np.float64)
        self.module.valmod.addd(0.5, x)
        assert list(x) == [1.5, 2.5]

    def test_c_int_value_function(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        assert self.module.valmod.sumi(10, x) == 16
