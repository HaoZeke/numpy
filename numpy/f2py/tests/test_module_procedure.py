import pytest

from . import util


@pytest.mark.slow
class TestModuleProcedure(util.F2PyTest):
    # The body is a submodule `module procedure`. The interface in the
    # parent is what Python calls.
    suffix = ".f90"
    code = """
module m
  interface
    module function ramp(n) result(y)
      integer, intent(in) :: n
      real :: y
    end function ramp
  end interface
end module m
submodule (m) mimpl
contains
  module procedure ramp
    y = n
  end procedure ramp
end submodule mimpl
"""

    def test_module_procedure_is_callable(self):
        assert self.module.m.ramp(4) == 4
