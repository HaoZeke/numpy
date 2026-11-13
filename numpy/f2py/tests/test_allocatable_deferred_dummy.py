import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAllocatableDeferredDummy(util.F2PyTest):
    # allocatable dimension(:) is deferred. An explicit-shape stand-in is
    # not a legal actual argument, for a numeric array or a deferred-length
    # character array.
    suffix = ".f90"
    code = """
module m
contains
  subroutine bounds(n, first, last)
    integer, intent(in) :: n
    integer, allocatable, intent(out) :: first(:), last(:)
    allocate(first(n), last(n))
    first = 1
    last = n
  end subroutine bounds
  subroutine grow(n, values)
    integer, intent(in) :: n
    real, allocatable, intent(out) :: values(:)
    integer :: i
    allocate(values(n))
    do i = 1, n
      values(i) = real(i)
    end do
  end subroutine grow
  subroutine tokens(line, array)
    character(len=*), intent(in) :: line
    character(len=:), allocatable, intent(out) :: array(:)
    allocate(character(len=len(line)) :: array(2))
    array(1) = line
    array(2) = line
  end subroutine tokens
end module m
"""

    def test_integer_and_real_and_character(self):
        first, last = self.module.m.bounds(3)
        assert list(first) == [1, 1, 1]
        assert list(last) == [3, 3, 3]
        values = np.asarray(self.module.m.grow(2))
        assert list(values) == [1.0, 2.0]
        got = self.module.m.tokens("Ab")
        assert [bytes(item).rstrip(b"\x00").decode() if not isinstance(item, str)
                else item.strip() for item in np.asarray(got)] == ["Ab", "Ab"]
