import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestProcedureDummy(util.F2PyTest):
    # procedure(iface) is a callback. The abstract interface supplies
    # the signature. On main this dummy has no C mapping and the module
    # is not wrapped.
    suffix = ".f90"
    code = """
module m
  abstract interface
    subroutine addone(n, x)
      integer, intent(in) :: n
      real, intent(inout) :: x(n)
    end subroutine addone
    real function twice(x)
      real, intent(in) :: x
    end function twice
  end interface
contains
  subroutine apply_sub(fcn, n, x)
    procedure(addone) :: fcn
    integer, intent(in) :: n
    real, intent(inout) :: x(n)
    call fcn(n, x)
  end subroutine apply_sub
  function apply_fun(f, x) result(y)
    procedure(twice) :: f
    real, intent(in) :: x
    real :: y
    y = f(x)
  end function apply_fun
end module
"""

    def test_subroutine_and_function_callbacks(self):
        def addone(x):
            x[0] = x[0] + 1

        arr = np.array([1.0, 2.0], dtype=np.float32)
        # n is the shape of x, so f2py hides it.
        self.module.m.apply_sub(addone, arr)
        assert arr[0] == np.float32(2.0)

        def twice(x):
            return 2.0 * x

        got = self.module.m.apply_fun(twice, np.float32(3.0))
        assert abs(float(got) - 6.0) < 1e-5
