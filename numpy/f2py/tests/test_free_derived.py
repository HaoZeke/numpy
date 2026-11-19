import pytest

from . import util


@pytest.mark.slow
class TestFreeDerived(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine scale(p, y)
  use iso_c_binding
  implicit none
  type, bind(c) :: vec
    integer(c_int) :: n
    real(c_float) :: data(4)
  end type vec
  type(vec), intent(inout) :: p
  integer, intent(out) :: y
  y = p%n
end subroutine scale

subroutine value_scale(p, y)
  use iso_c_binding
  implicit none
  type, bind(c) :: vec
    integer(c_int) :: n
  end type vec
  type(vec), value, intent(in) :: p
  integer, intent(out) :: y
  y = p%n
end subroutine value_scale

subroutine add(k, y)
  implicit none
  integer, intent(in) :: k
  integer, intent(out) :: y
  y = k + 1
end subroutine add

function make(n) result(p)
  implicit none
  type :: buf
    integer, allocatable :: data(:)
  end type buf
  integer, intent(in) :: n
  type(buf) :: p
  allocate(p%data(1))
  p%data(1) = n
end function make

function inc(k) result(y)
  implicit none
  integer, intent(in) :: k
  integer :: y
  y = k + 3
end function inc
"""

    @pytest.mark.parametrize("kept, arg, expected", [
        ("add", 3, 4),
        ("inc", 3, 6),
    ])
    def test_sibling_stays(self, kept, arg, expected):
        assert getattr(self.module, kept)(arg) == expected
        assert not hasattr(self.module, "scale")
        assert not hasattr(self.module, "value_scale")
        assert not hasattr(self.module, "make")
