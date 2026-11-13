import pytest

from . import util


@pytest.mark.slow
class TestDerivedTypeSkipped(util.F2PyTest):
    # A derived type has no C mapping. Building the module used to
    # raise KeyError and drop the routines that do have a mapping.
    suffix = ".f90"
    code = """
module m
  type :: pair
    integer :: a
  end type
  type(pair) :: saved
  integer, parameter :: n = 3
contains
  subroutine touch(p)
    type(pair), intent(in) :: p
  end subroutine
  function add(k) result(y)
    integer, intent(in) :: k
    integer :: y
    y = k + n
  end function
end module
"""

    def test_scalar_routine_is_wrapped(self):
        assert self.module.m.add(1) == 4
        assert self.module.m.n == 3

    def test_derived_routine_is_absent(self):
        assert not hasattr(self.module.m, "touch")
        assert not hasattr(self.module.m, "saved")
