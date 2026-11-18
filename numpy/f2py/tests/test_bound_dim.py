import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestBoundDim(util.F2PyTest):
    suffix = ".f90"
    code = """
      module bound_dim
      contains
        subroutine span_explicit(n, a, b)
          integer, intent(in) :: n
          integer, intent(in) :: a(0:n)
          integer, intent(out) :: b(lbound(a, 1):ubound(a, 1))
          b = a
        end subroutine span_explicit

        subroutine span_assumed(a, b)
          integer, intent(in) :: a(:)
          integer, intent(out) :: b(lbound(a, 1):ubound(a, 1))
          b = a
        end subroutine span_assumed
      end module bound_dim
    """

    @pytest.mark.parametrize("routine", ["span_explicit", "span_assumed"])
    def test_bound_pair(self, routine):
        a = np.array([3, 1, 4], dtype=np.int32)
        got = getattr(self.module.bound_dim, routine)(a)
        np.testing.assert_array_equal(got, a)
