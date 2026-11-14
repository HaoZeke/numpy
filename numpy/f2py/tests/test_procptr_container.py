import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestProcPtrContainer(util.F2PyTest):
    # A derived type that only holds a procedure pointer is the
    # callback. The wrapper builds the type and points the component
    # at the Python function.
    suffix = ".f90"
    code = """
module m
  implicit none
  abstract interface
    real function fcn(x)
      real, intent(in) :: x
    end function fcn
  end interface
  type container
    procedure(fcn), nopass, pointer :: f => null()
  end type container
contains
  subroutine apply(obj, x, y)
    type(container), intent(in) :: obj
    real, intent(in) :: x
    real, intent(out) :: y
    y = obj%f(x)
  end subroutine apply
end module m
"""

    def test_python_callback_through_the_type(self):
        def addone(x):
            return x + 1.0

        got = self.module.m.apply(addone, np.float32(3.0))
        assert abs(float(got) - 4.0) < 1e-5
