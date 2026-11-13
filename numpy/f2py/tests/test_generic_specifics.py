import pytest

from . import util


@pytest.mark.slow
class TestGenericSpecifics(util.F2PyTest):
    # Every specific of a public generic is callable. A long interface
    # used to keep only the last `module procedure`.
    suffix = ".f90"
    code = """
module m
  private
  public :: add
  interface operator(.add.)
    module procedure add_i
  end interface
  interface add
    module function add_i(a, b) result(c)
      integer, intent(in) :: a, b
      integer :: c
    end function add_i
    module function add_r(a, b) result(c)
      real, intent(in) :: a, b
      real :: c
    end function add_r
    module function add_d(a, b) result(c)
      real(8), intent(in) :: a, b
      real(8) :: c
    end function add_d
  end interface add
end module m
submodule (m) mimpl
contains
  module procedure add_i
    c = a + b
  end procedure add_i
  module procedure add_r
    c = a + b
  end procedure add_r
  module procedure add_d
    c = a + b
  end procedure add_d
end submodule mimpl
"""

    def test_every_specific_is_callable(self):
        assert self.module.m.add_i(2, 3) == 5
        assert self.module.m.add_r(2.0, 3.0) == 5
        assert self.module.m.add_d(2.0, 3.0) == 5
