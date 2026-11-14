import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestProcPtrCPtr(util.F2PyTest):
    # The procedure pointer's interface has ``type(c_ptr), value``.
    # The wrapper must give the callback an explicit interface, or the
    # pointer assignment is rejected.
    suffix = ".f90"
    code = """
module m
  use iso_c_binding, only: c_ptr, c_loc, c_null_ptr
  implicit none
  abstract interface
    subroutine fcn(n, data)
      import :: c_ptr
      integer, intent(in) :: n
      type(c_ptr), intent(in), value :: data
    end subroutine fcn
  end interface
  type box
    procedure(fcn), nopass, pointer :: f => null()
  end type box
contains
  subroutine apply(obj)
    type(box), intent(in) :: obj
    integer, target :: tag
    tag = 7
    call obj%f(1, c_null_ptr)
    call obj%f(2, c_loc(tag))
  end subroutine apply
end module m
"""

    def test_null_and_address_through_the_type(self):
        seen = []

        def f(n, data):
            seen.append((int(n), data))

        self.module.m.apply(f)
        assert len(seen) == 2
        assert seen[0] == (1, None)
        assert seen[1][0] == 2
        assert isinstance(seen[1][1], int)
        assert seen[1][1] > 0
