import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestMinvalShape(util.F2PyTest):
    # The diagonal length is minval(shape(a)). That has to be a C
    # expression; `minval` and `shape` are not C.
    suffix = ".f90"
    code = """
function diag(a) result(res)
  real, intent(in) :: a(:,:)
  real :: res(minval(shape(a)))
  integer :: i
  do i = 1, minval(shape(a))
    res(i) = a(i, i)
  end do
end function
"""

    def test_diagonal_length_is_the_smaller_extent(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = self.module.diag(np.asfortranarray(a))
        assert len(out) == 2
        assert out[0] == np.float32(1.0)
