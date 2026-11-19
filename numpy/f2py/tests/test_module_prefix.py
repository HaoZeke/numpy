import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestModulePrefix(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
contains
  module pure subroutine addtwo(a, b)
    integer, intent(in) :: a
    integer, intent(inout) :: b
    b = a + b
  end subroutine addtwo

  module recursive subroutine bump(n)
    integer, intent(inout) :: n
    n = n + 1
  end subroutine bump
end module m
"""

    @pytest.mark.parametrize("name, init, extra, expected", [
        ("addtwo", 4, (3,), 7),
        ("bump", 3, (), 4),
    ])
    def test_module_prefix(self, name, init, extra, expected):
        out = np.array([init], dtype=np.int32)
        getattr(self.module.m, name)(*extra, out)
        assert int(out[0]) == expected
