import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestAuxDepend(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine scale_aux(a, b, n)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: b(n)
  !f2py integer, intent(aux), depend(a) :: k = len(a)
  !f2py integer, intent(hide), depend(k) :: n = k
  b = a * 2.0d0
end subroutine scale_aux

subroutine offset_aux(x, y)
  integer, intent(in) :: x
  integer, intent(out) :: y
  !f2py integer, intent(aux, out), depend(x), check(z > 0) :: z = x + 7
  y = x * 2
end subroutine offset_aux

subroutine chain_aux(x, y)
  integer, intent(in) :: x
  integer, intent(out) :: y
  !f2py integer, intent(aux), depend(x) :: k1 = x + 1
  !f2py integer, intent(aux, out), depend(k1) :: k2 = 2 * k1
  y = x
end subroutine chain_aux

subroutine limit_aux(j, y)
  integer, intent(in) :: j
  integer, intent(out) :: y
  !f2py integer, intent(aux) :: lim = 5
  !f2py check(j < lim) :: j
  y = j
end subroutine limit_aux
"""

    def test_aux_initialized_after_dependencies(self):
        out = self.module.scale_aux(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(out, [2.0, 4.0, 6.0])

    def test_aux_check_message(self):
        assert self.module.offset_aux(5) == (12, 10)
        with pytest.raises(Exception, match="auxiliary z"):
            self.module.offset_aux(-15)

    def test_aux_chain(self):
        assert self.module.chain_aux(5) == (12, 5)

    def test_independent_aux_visible_to_checks(self):
        assert self.module.limit_aux(3) == 3
        with pytest.raises(Exception, match="failed for"):
            self.module.limit_aux(7)
