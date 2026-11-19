import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestResultModKind(util.F2PyTest):
    # The result name is not the function name. The kind is a parameter
    # of the module that contains the function.
    suffix = ".f90"
    code = """
module kmod
  implicit none
  integer, parameter :: ik = selected_int_kind(18)
contains
  integer(kind=ik) function add_prefix(n) result(r)
    implicit none
    integer(kind=ik), intent(in) :: n
    r = n + 1
  end function add_prefix

  function add_decl(n) result(r)
    implicit none
    integer(kind=ik), intent(in) :: n
    integer(kind=ik) :: r
    r = n + 1
  end function add_decl
end module kmod
"""

    @pytest.mark.parametrize("name", ["add_prefix", "add_decl"])
    def test_wide_result(self, name):
        fn = getattr(self.module.kmod, name)
        base = np.int64(2 ** 40)
        assert fn(base) == int(base) + 1
