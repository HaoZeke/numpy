import textwrap

import pytest

import numpy as np

from . import util


class TestReversedBound(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent("""
        module revbound
          implicit none
        contains
          subroutine fill_low(a, n, s)
            integer, intent(in) :: n
            integer, intent(out) :: a(0:n)
            integer, intent(out) :: s
            integer :: i
            s = size(a)
            do i = 0, n
              a(i) = i
            end do
          end subroutine fill_low

          subroutine count_rev(a, n, s)
            integer, intent(in) :: n
            integer, intent(in) :: a(n:0)
            integer, intent(out) :: s
            s = size(a)
          end subroutine count_rev
        end module revbound
        """)

    @pytest.mark.parametrize(
        ("n", "low_size", "rev_size"),
        [(-2, 0, 3), (2, 3, 0)],
    )
    def test_extent(self, n, low_size, rev_size):
        a, s = self.module.revbound.fill_low(n)
        assert s == low_size
        assert a.shape == (low_size,)
        assert a.tolist() == list(range(low_size))
        got = self.module.revbound.count_rev(
            np.zeros(rev_size, dtype=np.intc), n
        )
        assert got == rev_size
