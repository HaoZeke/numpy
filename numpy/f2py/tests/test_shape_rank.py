import numpy as np
import pytest

from . import util


class TestSizeShapeRank(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine rank1(x, y)
  real, intent(in) :: x(:)
  integer, intent(out) :: y(size(shape(x)))
  y = size(shape(x))
end subroutine rank1

subroutine rank2(x, y)
  real, intent(in) :: x(:,:)
  integer, intent(out) :: y(size(shape(x)))
  y = size(shape(x))
end subroutine rank2
"""

    @pytest.mark.parametrize("name, shape", [("rank1", (3,)), ("rank2", (2, 4))])
    def test_extent_is_rank(self, name, shape):
        x = np.zeros(shape, dtype=np.float32)
        y = getattr(self.module, name)(x)
        assert y.shape == (len(shape),)
        assert np.all(y == len(shape))
