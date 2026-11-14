import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestOptionalArrayDim(util.F2PyTest):
    # `extra(:)` is optional. Omitting it used to abort because the
    # extent was -1.
    suffix = ".f90"
    code = """
module m
contains
  integer function saw(x, extra) result(n)
    real, intent(in) :: x(:)
    real, intent(in), optional :: extra(:)
    if (present(extra)) then
      n = size(extra)
    else
      n = -1
    end if
  end function
end module
"""

    def test_omitted_optional_array_is_absent(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # A present zero-size array has size 0. Absent is -1.
        assert self.module.m.saw(x) == -1

    def test_passed_optional_array_is_used(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        extra = np.array([9.0, 8.0], dtype=np.float32)
        assert self.module.m.saw(x, extra) == 2
