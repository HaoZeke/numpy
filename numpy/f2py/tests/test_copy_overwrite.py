import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCopyOverwrite(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine copy_then_overwrite(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  !f2py intent(in, out, copy) :: a
  !f2py intent(overwrite) :: a
  a = a * 2.0d0
end subroutine copy_then_overwrite

subroutine overwrite_then_copy(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  !f2py intent(in, out, overwrite) :: a
  !f2py intent(copy) :: a
  a = a * 2.0d0
end subroutine overwrite_then_copy

subroutine only_overwrite(a, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(n)
  !f2py intent(in, out, overwrite) :: a
  a = a * 2.0d0
end subroutine only_overwrite

subroutine fill_out(n, a)
  integer, intent(in) :: n
  real(8), intent(out) :: a(n)
  !f2py intent(out, copy, overwrite) :: a
  a = 3.0d0
end subroutine fill_out
"""

    @pytest.mark.parametrize("name", ["copy_then_overwrite",
                                      "overwrite_then_copy"])
    def test_copy_takes_precedence(self, name):
        func = getattr(self.module, name)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64, order="F")
        out = func(a)
        assert np.allclose(out, [2.0, 4.0, 6.0])
        assert np.allclose(a, [1.0, 2.0, 3.0])
        out = func(a, overwrite_a=1)
        assert np.allclose(out, [2.0, 4.0, 6.0])
        assert np.allclose(a, [2.0, 4.0, 6.0])

    def test_overwrite_default_unchanged(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64, order="F")
        self.module.only_overwrite(a)
        assert np.allclose(a, [2.0, 4.0, 6.0])
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64, order="F")
        self.module.only_overwrite(a, overwrite_a=0)
        assert np.allclose(a, [1.0, 2.0, 3.0])

    def test_hidden_output_accepts_overwrite(self):
        assert np.allclose(self.module.fill_out(3), [3.0, 3.0, 3.0])
        assert np.allclose(self.module.fill_out(3, overwrite_a=1),
                           [3.0, 3.0, 3.0])
