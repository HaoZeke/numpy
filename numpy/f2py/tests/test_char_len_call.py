import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCharLenCall(util.F2PyTest):
    # The result length is a Fortran function. That call must not be
    # emitted as C.
    suffix = ".f90"
    code = """
module m
contains
  pure integer function nlen(i)
    integer, intent(in) :: i
    if (i < 0) then
      nlen = 2
    else if (abs(i) < 10) then
      nlen = 1
    else
      nlen = 2
    end if
  end function nlen
  function label(i) result(s)
    integer, intent(in) :: i
    character(len=nlen(i)) :: s
    write(s, '(i0)') i
  end function label
end module m
"""

    def test_label(self):
        assert self.module.m.label(7) == b"7"
        assert self.module.m.label(42) == b"42"
        assert self.module.m.label(-3) == b"-3"
