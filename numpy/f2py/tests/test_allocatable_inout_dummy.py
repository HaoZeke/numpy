import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAllocatableInoutDummy(util.F2PyTest):
    # `output(:)` is allocatable and intent(inout). An explicit-shape
    # stand-in is not a legal actual argument, so the module used to
    # fail to compile, including the sibling routine.
    suffix = ".f90"
    code = """
module m
contains
  integer function plain() result(n)
    n = 1
  end function plain
  subroutine keep(array, output)
    integer, intent(in) :: array(:)
    integer, allocatable, intent(inout) :: output(:)
    if (allocated(output)) deallocate(output)
    allocate(output(size(array)))
    output = array + 1
  end subroutine keep
end module m
"""

    def test_sibling_and_allocatable_inout(self):
        assert self.module.m.plain() == 1
        array = np.array([1, 2, 3], dtype=np.int32)
        got = np.asarray(self.module.m.keep(array))
        assert list(got) == [2, 3, 4]
