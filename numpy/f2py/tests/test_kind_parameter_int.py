import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestKindParameterInt(util.F2PyTest):
    # dp = selected_real_kind(15) is 8. The wrapper must use that
    # width, including when the kind module is not the parent that
    # f2py wraps the sources in.
    suffix = ".f90"
    code = """
module kinds
  integer, parameter :: dp = selected_real_kind(15)
end module kinds
module bridge
  use kinds, only: dp
end module bridge
module m
  use bridge, only: dp
contains
  subroutine twice(x)
    real(dp), intent(inout) :: x
    x = x * 2
  end subroutine twice
end module m
"""

    def test_double_precision(self):
        x = np.array([1.5], dtype=np.float64)
        self.module.m.twice(x)
        assert x.dtype == np.dtype(np.float64)
        assert x[0] == np.float64(3.0)
