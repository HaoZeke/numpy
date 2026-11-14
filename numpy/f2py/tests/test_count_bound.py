import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCountBound(util.F2PyTest):
    # ``integer :: loc(count(array))`` used to be copied into C as a
    # call to an undeclared ``count``.
    suffix = ".f90"
    code = """
module m
contains
  function trueloc(array) result(loc)
    logical, intent(in) :: array(:)
    integer :: loc(count(array))
    integer :: i, n
    n = 0
    do i = 1, size(array)
      if (array(i)) then
        n = n + 1
        loc(n) = i
      end if
    end do
  end function
  function falseloc(array) result(loc)
    logical, intent(in) :: array(:)
    integer :: loc(count(.not. array))
    integer :: i, n
    n = 0
    do i = 1, size(array)
      if (.not. array(i)) then
        n = n + 1
        loc(n) = i
      end if
    end do
  end function
end module
"""

    def test_true_and_false_locations(self):
        mask = np.array([True, False, True, True])
        assert list(self.module.m.trueloc(mask)) == [1, 3, 4]
        assert list(self.module.m.falseloc(mask)) == [2]
