import pytest

from . import util


@pytest.mark.slow
class TestPrivateSpecific(util.F2PyTest):
    # The specific is private. `use m, only: add_i` is illegal.
    # The wrapper has to call the public generic.
    suffix = ".f90"
    code = """
module m
  private
  public :: add, plain
  interface add
    module procedure add_i
  end interface
contains
  subroutine add_i(n, k, out)
    integer, intent(in) :: n, k
    integer, intent(out) :: out
    out = n + k
  end subroutine
  integer function plain()
    plain = 1
  end function
end module
"""

    def test_generic_wrapper_and_sibling(self):
        assert self.module.m.plain() == 1
        assert self.module.m.add_i(1, 2) == 3
