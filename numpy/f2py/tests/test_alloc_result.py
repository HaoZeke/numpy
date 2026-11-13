import pytest

from . import util


@pytest.mark.slow
class TestAllocatableResult(util.F2PyTest):
    # The result is allocated by Fortran. Python must receive that array.
    suffix = ".f90"
    code = """
module m
contains
  function ramp(n) result(y)
    integer, intent(in) :: n
    real, allocatable :: y(:)
    integer :: i
    allocate(y(n))
    do i = 1, n
      y(i) = i
    end do
  end function ramp
  function grid(n) result(y)
    integer, intent(in) :: n
    real, allocatable :: y(:,:)
    integer :: i, j
    allocate(y(n, 2))
    do j = 1, 2
      do i = 1, n
        y(i, j) = 10 * i + j
      end do
    end do
  end function grid
end module m
"""

    def test_rank1_allocatable_result(self):
        got = self.module.m.ramp(3)
        assert list(got) == [1.0, 2.0, 3.0]

    def test_rank2_allocatable_result(self):
        got = self.module.m.grid(2)
        assert got.shape == (2, 2)
        assert got[0, 0] == 11
        assert got[1, 1] == 22
