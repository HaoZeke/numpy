import pytest

from . import util


@pytest.mark.slow
class TestPublicUseOnly(util.F2PyTest):
    # `rot` is local to the submodule. `use math, only : rot` is not
    # a public entity and used to reject the wrapper.
    suffix = ".f90"
    code = """
module math
  implicit none
  interface
    module function ramp(n) result(y)
      integer, intent(in) :: n
      integer :: y
    end function ramp
  end interface
end module math
submodule (math) simpl
contains
  module function ramp(n) result(y)
    integer, intent(in) :: n
    integer :: y
    y = n + 1
  end function ramp
  integer function rot(a)
    integer, intent(in) :: a
    rot = a
  end function rot
end submodule simpl
"""

    def test_public_procedure_is_callable(self):
        assert self.module.math.ramp(4) == 5
