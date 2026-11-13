import pytest

from . import util


@pytest.mark.slow
class TestPrivateSpecific(util.F2PyTest):
    # `hidden` is not a public symbol of `parent`. Importing it with
    # `use parent, only : hidden` must not be emitted, or the public
    # `work` procedure fails to build.
    suffix = ".f90"
    code = """
module parent
  implicit none
  public :: work
  interface
    module function work(n) result(y)
      integer, intent(in) :: n
      integer :: y
    end function work
  end interface
end module parent
submodule (parent) child
contains
  module function work(n) result(y)
    integer, intent(in) :: n
    integer :: y
    y = n + 1
  end function work
  subroutine hidden(n)
    integer, intent(out) :: n
    n = 0
  end subroutine hidden
end submodule child
"""

    def test_public_procedure_builds_without_private_import(self):
        assert self.module.parent.work(3) == 4
