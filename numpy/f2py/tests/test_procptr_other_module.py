import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestProcPtrOtherModule(util.F2PyTest):
    # The procedure pointer's abstract interface lives in the module
    # that defines the type, not in the module that takes the dummy.
    suffix = ".f90"
    code = """
module defs
  implicit none
  abstract interface
    real function fcn(x)
      real, intent(in) :: x
    end function fcn
  end interface
  type box
    procedure(fcn), nopass, pointer :: f => null()
  end type box
end module defs
module useit
  use defs, only: box
  implicit none
contains
  subroutine apply(obj, x, y)
    type(box), intent(in) :: obj
    real, intent(in) :: x
    real, intent(out) :: y
    y = obj%f(x)
  end subroutine apply
end module useit
"""

    def test_callback_whose_interface_is_in_another_module(self):
        def addone(x):
            return x + 1.0

        got = self.module.useit.apply(addone, np.float32(3.0))
        assert abs(float(got) - 4.0) < 1e-5
