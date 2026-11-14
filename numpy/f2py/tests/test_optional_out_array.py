import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestOptionalOutArray(util.F2PyTest):
    # `work(:)` is optional and intent(out). Omitting it used to
    # allocate an array with no extent, so `present(work)` was true
    # and the write crashed. An omitted dummy is absent.
    suffix = ".f90"
    code = """
module m
contains
  subroutine bump(array, work)
    integer, intent(inout) :: array(:)
    integer, intent(out), optional :: work(:)
    integer :: i
    do i = 1, size(array)
      array(i) = array(i) + 1
    end do
    if (present(work)) then
      work = 7
    end if
  end subroutine
end module
"""

    def test_omitted_optional_out_array_is_absent(self):
        array = np.array([1, 2, 3], dtype=np.int32)
        got = self.module.m.bump(array)
        assert got is None
        assert list(array) == [2, 3, 4]
