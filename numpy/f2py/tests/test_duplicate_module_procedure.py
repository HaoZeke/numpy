import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestDuplicateModuleProcedure(util.F2PyTest):
    # The interface declaration and the contains implementation are one
    # procedure. Wrapping both defines the same C function twice.
    suffix = ".f90"
    code = """
module dupmod
  interface
    pure module function twice(n) result(y)
      integer, intent(in) :: n
      integer :: y
    end function twice
  end interface
contains
  pure module function twice(n) result(y)
    integer, intent(in) :: n
    integer :: y
    y = n + n
  end function twice
end module dupmod
"""

    def test_interface_and_implementation_are_one_wrapper(self):
        assert self.module.dupmod.twice(np.int32(3)) == 6
