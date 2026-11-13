import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestElementalSubroutine(util.F2PyTest):
    # gfortran rejects an elemental subroutine passed as an actual
    # argument. The wrapper has to be an ordinary subroutine.
    suffix = ".f90"
    code = """
module m
contains
  elemental subroutine addone(x)
    integer, intent(inout) :: x
    x = x + 1
  end subroutine addone
end module m
"""

    def test_addone(self):
        x = np.array([3], dtype=np.int32)
        self.module.m.addone(x)
        assert int(x[0]) == 4
