import pytest

from . import util


@pytest.mark.slow
class TestNamedCheck(util.F2PyTest):
    # A procedure named `check` is not a check= clause.
    suffix = ".f90"
    code = """
module m
  public :: check
contains
  function check(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 1
  end function check
end module m
"""

    def test_procedure_named_check(self):
        assert self.module.m.check(3) == 4
