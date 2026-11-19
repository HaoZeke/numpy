import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestSelectedRealKindRange(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine by_range(x)
        integer, parameter :: rk = selected_real_kind(6, 300)
        real(kind=rk), intent(out) :: x
        x = 1.5_rk
      end subroutine by_range

      subroutine by_name(x)
        integer, parameter :: rk = selected_real_kind(r=307)
        real(kind=rk), intent(out) :: x
        x = 1.5_rk
      end subroutine by_name
    """

    @pytest.mark.parametrize("name", ["by_range", "by_name"])
    def test_range_selects_double(self, name):
        got = getattr(self.module, name)()
        assert got == np.float64(1.5)
