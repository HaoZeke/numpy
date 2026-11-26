import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestComplexOptionalDefault(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine pick(a, b, out)
  complex(kind=4), intent(in) :: a
  complex(kind=4), intent(in) :: b
  complex(kind=4), intent(out) :: out
  !f2py complex(kind=4), optional, intent(in), depend(a), check(b.r>0) :: b = a
  out = b
end subroutine pick
"""

    @pytest.mark.parametrize("override", [None, 3 + 4j])
    def test_default_names_other_argument(self, override):
        a = 1 + 2j
        if override is None:
            out = self.module.pick(np.complex64(a))
            expect = a
        else:
            out = self.module.pick(np.complex64(a), np.complex64(override))
            expect = override
        assert complex(out) == complex(expect)
