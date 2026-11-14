import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestOptionalAbsent(util.F2PyTest):
    # An omitted optional is not present. Passing it as zero makes
    # `arange(1.5)` look like `arange(1.5, 0)`.
    suffix = ".f90"
    code = """
module m
contains
  real function widen(start, end) result(y)
    real, intent(in) :: start
    real, intent(in), optional :: end
    if (present(end)) then
      y = end
    else
      y = start
    end if
  end function
end module
"""

    def test_omitted_optional_is_absent(self):
        assert self.module.m.widen(np.float32(1.5)) == np.float32(1.5)
