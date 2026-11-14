import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestOptionalComplexAbsent(util.F2PyTest):
    # An omitted optional complex is not present. Passing the
    # uninitialized local makes `present` true and the result garbage.
    suffix = ".f90"
    code = """
module m
contains
  complex function pick(x, fallback) result(y)
    complex, intent(in), optional :: x
    complex, intent(in) :: fallback
    if (present(x)) then
      y = x
    else
      y = fallback
    end if
  end function
end module
"""

    def test_omitted_optional_complex_is_absent(self):
        got = self.module.m.pick(np.complex64(1 + 0.5j))
        assert got == np.complex64(1 + 0.5j)
