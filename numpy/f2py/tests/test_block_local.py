import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestBlockLocal(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine scale(x)
  implicit none
  real, intent(inout) :: x
  outer: block
    real :: x(2)
    x(1) = 1.0
    x(2) = 2.0
  endblock outer
  x = x + 1.0
end subroutine scale

subroutine after(y)
  implicit none
  real, intent(inout) :: y
  y = y + 1.0
end subroutine after
"""

    def test_scalar_not_reshaped(self):
        x = np.array(1.5, dtype=np.float32)
        self.module.scale(x)
        assert x.item() == np.float32(2.5)
        y = np.array(3.0, dtype=np.float32)
        self.module.after(y)
        assert y.item() == np.float32(4.0)
