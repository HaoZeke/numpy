import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestUseKind(util.F2PyTest):
    # ``rk => real64`` is a kind, including through another module and
    # ``use, intrinsic``. It used to stay the name ``rk`` and become
    # default real.
    suffix = ".f90"
    code = """
module kinds
  use, intrinsic :: iso_fortran_env, only: rk => real64
end module kinds
module fft
  use kinds, only: rk
  use, intrinsic :: iso_fortran_env, only: wp => real64
contains
  function widen(x) result(y)
    real(kind=rk), intent(in) :: x
    real(kind=rk) :: y
    y = x + 1.0_rk
  end function widen
  function widen_wp(x) result(y)
    real(kind=wp), intent(in) :: x
    real(kind=wp) :: y
    y = x + 1.0_wp
  end function widen_wp
end module fft
"""

    def test_renamed_real64_is_double(self):
        got = np.asarray(self.module.fft.widen(np.float64(1.0)))
        assert got.dtype == np.float64
        assert got == np.float64(2.0)
        got = np.asarray(self.module.fft.widen_wp(np.float64(1.0)))
        assert got.dtype == np.float64
        assert got == np.float64(2.0)
