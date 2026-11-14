import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAssumedOutExtent(util.F2PyTest):
    # `index(:)` is a required intent(out) array with no extent.
    # One earlier input of the same rank supplies it. Two inputs do
    # not, and the caller has to pass the array.
    suffix = ".f90"
    code = """
module m
contains
  subroutine fill(array, index)
    integer, intent(inout) :: array(:)
    integer, intent(out) :: index(:)
    integer :: i
    do i = 1, size(array)
      index(i) = array(i) + 1
    end do
  end subroutine fill
  subroutine pick(left, right, index)
    integer, intent(in) :: left(:), right(:)
    integer, intent(out) :: index(:)
    index = left + right
  end subroutine pick
end module
"""

    def test_extent_comes_from_the_input(self):
        array = np.array([1, 2, 3], dtype=np.int32)
        got = np.asarray(self.module.m.fill(array))
        assert list(got) == [2, 3, 4]

    def test_ambiguous_extent_requires_the_array(self):
        left = np.array([1, 2], dtype=np.int32)
        right = np.array([3, 4], dtype=np.int32)
        with pytest.raises(Exception, match="no extent"):
            self.module.m.pick(left, right)
        index = np.zeros(2, dtype=np.int32)
        self.module.m.pick(left, right, index)
        assert list(index) == [4, 6]
