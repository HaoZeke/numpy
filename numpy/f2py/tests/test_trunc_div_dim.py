import pytest

from . import util


class TestTruncDivDim(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine fill_half(a, n, m)
  integer, intent(in) :: n, m
  integer, intent(out) :: a((n/2)*m)
end subroutine fill_half
"""

    @pytest.mark.parametrize(("n", "m", "size"), [(5, 3, 6)])
    def test_extent(self, n, m, size):
        a = self.module.fill_half(n, m)
        assert a.shape == (size,)
