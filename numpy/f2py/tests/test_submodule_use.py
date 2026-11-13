import pytest

from . import util


@pytest.mark.slow
class TestSubmoduleUse(util.F2PyTest):
    # The implementation is a submodule procedure. `use simpl` cannot
    # open `simpl.mod`. The call goes through the parent, and the
    # wrapper stays `f2pywrap_simpl_ramp` so it does not collide with
    # `f2pywrap_math_ramp`.
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
    y = n
  end function ramp
end submodule simpl
"""

    def test_submodule_procedure_is_called_through_parent(self):
        assert self.module.math.ramp(3) == 3
