import pytest

import numpy as np

from . import util


def _trunc_div(numer, denom):
    return (abs(numer) // abs(denom)
            * (-1 if (numer < 0) != (denom < 0) else 1))


@pytest.mark.slow
class TestDivExtent(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine take_plus_out(a, b, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n + 1) / 2), intent(in) :: a
  integer, dimension(n), intent(out) :: b
  b = 1
end subroutine

subroutine take_lower_zero(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension(0:(n - 3) / 4), intent(in) :: a
end subroutine

subroutine take_lower_two(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension(2:(n + 1) / 2), intent(in) :: a
end subroutine

subroutine take_neg_offset(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension(10 - (n + 1) / 2), intent(in) :: a
end subroutine

subroutine take_neg_denom(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n + 1) / (-2)), intent(in) :: a
end subroutine

subroutine take_scale_offset(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((3 * n) / 2 + 1), intent(in) :: a
end subroutine

subroutine take_quarter(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n + 1) / 4), intent(in) :: a
end subroutine

subroutine take_scale_lo(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((3 * n - 7) / 2), intent(in) :: a
end subroutine

subroutine take_scale_hi(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((3 * n + 4) / 2), intent(in) :: a
end subroutine

subroutine take_even_shift(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n - 2) / 2), intent(in) :: a
end subroutine

subroutine take_zero_lo(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n - 5) / 2), intent(in) :: a
end subroutine

subroutine take_zero_hi(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n + 5) / 2), intent(in) :: a
end subroutine

subroutine take_two_quotients(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n + 1) / 2 + (n + 2) / 3), intent(in) :: a
end subroutine

subroutine take_matrix(a, m, n)
  integer :: m, n
  !f2py intent(out) m, n
  integer, dimension(m, (n + 1) / 2), intent(in) :: a
end subroutine

subroutine take_keep(a, n)
  integer :: n
  !f2py intent(out) n
  integer, dimension((n - 1) / 2), intent(in) :: a
end subroutine

subroutine check_offset_plus(a, n)
  integer, intent(in) :: n
  integer, dimension((n - 3) / 4 + 1), intent(in) :: a
end subroutine
"""

    @pytest.mark.parametrize("routine,size,extent", [
        ("take_lower_zero", 1, lambda n: _trunc_div(n - 3, 4) + 1),
        ("take_lower_two", 1, lambda n: _trunc_div(n + 1, 2) - 1),
        ("take_neg_offset", 0, lambda n: 10 - _trunc_div(n + 1, 2)),
        ("take_neg_denom", 1, lambda n: _trunc_div(n + 1, -2)),
        ("take_scale_offset", 0, lambda n: _trunc_div(3 * n, 2) + 1),
        ("take_quarter", 0, lambda n: _trunc_div(n + 1, 4)),
        ("take_scale_lo", 0, lambda n: _trunc_div(3 * n - 7, 2)),
        ("take_scale_hi", 0, lambda n: _trunc_div(3 * n + 4, 2)),
        ("take_even_shift", 1, lambda n: _trunc_div(n - 2, 2)),
        ("take_zero_lo", 0, lambda n: _trunc_div(n - 5, 2)),
        ("take_zero_hi", 0, lambda n: _trunc_div(n + 5, 2)),
    ])
    def test_hidden_size(self, routine, size, extent):
        a = np.ones(size, dtype=np.int32)
        assert extent(getattr(self.module, routine)(a)) == size

    def test_second_dimension(self):
        m, n = self.module.take_matrix(np.ones((3, 4), dtype=np.int32))
        assert m == 3
        assert _trunc_div(n + 1, 2) == 4

    def test_dependent_and_checks(self):
        b, n = self.module.take_plus_out(np.ones(3, dtype=np.int32))
        assert _trunc_div(n + 1, 2) == 3
        assert b.shape == (n,)
        assert self.module.take_keep(np.ones(0, dtype=np.int32)) == 1
        n = self.module.take_two_quotients(np.ones(2, dtype=np.int32))
        assert _trunc_div(n + 1, 2) + _trunc_div(n + 2, 3) == 2
        self.module.check_offset_plus(np.ones(2, dtype=np.int32))
        err = getattr(self.module, f"_{self.module.__name__}_error")
        with pytest.raises(err):
            self.module.take_scale_offset(np.ones(3, dtype=np.int32))
        with pytest.raises(err):
            self.module.check_offset_plus(np.ones(2, dtype=np.int32), n=0)
