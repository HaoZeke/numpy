import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAllocatableFunctionResult(util.F2PyTest):
    # The result is allocated in Fortran. The wrapper must not try to
    # build it first: that size is still -1 and the call raises.
    suffix = ".f90"
    code = """
function f(n) result(a)
  integer, intent(in) :: n
  integer, allocatable :: a(:)
  integer :: i
  if (n > 0) then
    allocate(a(n))
    do i = 1, n
      a(i) = n
    end do
  else
    allocate(a(0))
  end if
end function
"""

    def test_returns_allocated_array(self):
        got = np.asarray(self.module.f(3))
        assert got.shape == (3,)
        assert got.tolist() == [3, 3, 3]

    def test_empty_allocation(self):
        got = np.asarray(self.module.f(0))
        assert got.shape == (0,)
        assert got.tolist() == []
