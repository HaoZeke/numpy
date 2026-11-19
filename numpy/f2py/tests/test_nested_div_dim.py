import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestNestedDivDim(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine nested_div(n, m, k, a)
  integer, intent(in) :: n, m, k
  integer(kind=1), intent(out) :: a(n/(m/k), 2)
  a = 1_1
end subroutine nested_div
"""

    @pytest.mark.parametrize(("n", "m", "k", "extent"), [
        (10, 6, 4, 10),
        (7, 5, 3, 7),
    ])
    def test_extent(self, n, m, k, extent):
        a = self.module.nested_div(n, m, k)
        assert a.shape == (extent, 2)
        assert a.dtype == np.int8
