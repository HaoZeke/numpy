import pytest

from . import util


@pytest.mark.slow
class TestEndType(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
  type :: pair
    integer :: a
  endtype pair
contains
  subroutine add(k, y)
    integer, intent(in) :: k
    integer, intent(out) :: y
    y = k + 3
  end subroutine add
end module m
"""

    def test_routine_after_endtype_is_wrapped(self):
        assert self.module.m.add(1) == 4
