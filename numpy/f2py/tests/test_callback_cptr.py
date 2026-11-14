import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCallbackCPtr(util.F2PyTest):
    # ``type(c_ptr), value`` in a callback is an opaque address.
    # A null pointer is ``None``. Any other pointer is its integer address.
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
contains
  subroutine probe(f)
    procedure(fcn) :: f
    integer, target :: tag
    tag = 7
    call f(1, c_null_ptr)
    call f(2, c_loc(tag))
  end subroutine probe
end module m
"""

    def test_null_is_none_and_address_is_int(self):
        seen = []

        def f(n, data):
            seen.append((int(n), data))

        self.module.m.probe(f)
        assert len(seen) == 2
        assert seen[0][0] == 1
        assert seen[0][1] is None
        assert seen[1][0] == 2
        assert isinstance(seen[1][1], int)
        assert seen[1][1] > 0
