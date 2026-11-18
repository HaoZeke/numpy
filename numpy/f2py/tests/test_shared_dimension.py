import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestSharedDimension(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine twice_first(a, n, second)
        integer :: n
!f2py intent(out) n
        integer, intent(in) :: a(2 * n, n)
        integer, intent(out) :: second
        second = size(a, 2)
      end subroutine twice_first

      subroutine twice_second(a, n, second)
        integer :: n
!f2py intent(out) n
        integer, intent(in) :: a(n, 2 * n)
        integer, intent(out) :: second
        second = size(a, 2)
      end subroutine twice_second

      subroutine product(a, n, m, second)
        integer :: n
!f2py intent(out) n
        integer, intent(in) :: m
        integer, intent(in) :: a(n, n * m)
        integer, intent(out) :: second
        second = size(a, 2)
      end subroutine product
    """

    def test_inferred_dimension_checks(self):
        assert self.module.twice_first(
            np.zeros((6, 3), dtype=np.int32)
        ) == (3, 3)
        assert self.module.twice_second(
            np.zeros((3, 6), dtype=np.int32)
        ) == (3, 6)
        assert self.module.product(
            np.zeros((2, 6), dtype=np.int32), 3
        ) == (2, 6)
        with pytest.raises(Exception):
            self.module.twice_first(np.zeros((6, 4), dtype=np.int32))
        with pytest.raises(Exception):
            self.module.twice_second(np.zeros((3, 5), dtype=np.int32))
