import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCBool(util.F2PyTest):
    suffix = ".f90"
    code = """
module cboolmod
  use iso_c_binding, only: c_bool, c_int
  implicit none
contains
  subroutine outb(y)
    logical(c_bool), intent(out) :: y
    y = .true.
  end subroutine outb

  subroutine outbf(y)
    logical(c_bool), intent(out) :: y
    y = .false.
  end subroutine outbf

  subroutine inb(x, y)
    logical(c_bool), intent(in) :: x
    integer(c_int), intent(out) :: y
    if (x) then
      y = 7
    else
      y = 3
    end if
  end subroutine inb

  function isone(x) result(y)
    integer(c_int), intent(in) :: x
    logical(c_bool) :: y
    y = x == 1
  end function isone

  subroutine outbarr(y)
    logical(c_bool), intent(out) :: y(2)
    y(1) = .true.
    y(2) = .false.
  end subroutine outbarr

  subroutine inbarr(x, y)
    logical(c_bool), intent(in) :: x(2)
    integer(c_int), intent(out) :: y
    y = 0
    if (x(1)) y = y + 1
    if (x(2)) y = y + 2
  end subroutine inbarr
end module cboolmod
"""

    def test_scalar_out(self):
        mod = self.module.cboolmod
        assert mod.outb() == 1
        assert mod.outbf() == 0

    def test_scalar_in(self):
        mod = self.module.cboolmod
        assert mod.inb(True) == 7
        assert mod.inb(False) == 3

    def test_function(self):
        mod = self.module.cboolmod
        assert mod.isone(1) == 1
        assert mod.isone(0) == 0

    def test_array(self):
        mod = self.module.cboolmod
        out = np.asarray(mod.outbarr())
        assert out.dtype == np.uint8
        assert list(out) == [1, 0]
        got = mod.inbarr(np.array([1, 0], dtype=np.uint8))
        assert got == 1
        got = mod.inbarr(np.array([0, 1], dtype=np.uint8))
        assert got == 2
