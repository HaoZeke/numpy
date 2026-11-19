import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCCharKind(util.F2PyTest):
    suffix = ".f90"
    code = """
module cchar_kind
  use iso_c_binding, only: c_char
  implicit none
contains
  function echo(x) result(y)
    character(kind=c_char), intent(in) :: x
    character(kind=c_char) :: y
    y = x
  end function echo
  subroutine firstc(x, y)
    character(kind=c_char), intent(in) :: x(:)
    character(kind=c_char), intent(out) :: y
    y = x(1)
  end subroutine firstc
end module cchar_kind
"""

    @pytest.mark.parametrize(
        "name, value",
        [("echo", "Q"), ("firstc", np.array([b"Q"], dtype="S1"))],
    )
    def test_roundtrip(self, name, value):
        fn = getattr(self.module.cchar_kind, name)
        assert fn(value) == b"Q"
