import pytest

from . import util


@pytest.mark.slow
class TestPrivateGeneric(util.F2PyTest):
    # `bump` is a private generic. Its specific is not use-associated.
    # The wrapper is a submodule, so `bump_i` still runs.
    suffix = ".f90"
    code = """
module m
  private
  public :: shown
  interface bump
    module subroutine bump_i(n, y)
      integer, intent(in) :: n
      integer, intent(out) :: y
    end subroutine bump_i
  end interface bump
contains
  integer function shown() result(k)
    k = 1
  end function shown
end module m
submodule (m) m_impl
contains
  module subroutine bump_i(n, y)
    integer, intent(in) :: n
    integer, intent(out) :: y
    y = n + 1
  end subroutine bump_i
end submodule m_impl
"""

    def test_private_generic_specific_is_callable(self):
        assert self.module.m.shown() == 1
        assert self.module.m.bump_i(3) == 4
