import numpy as np
import pytest

from numpy.f2py.symbolic import Expr, Language

from . import util


def test_size_dim_keyword_is_positional_in_c():
    parsed = Expr.parse("size(a, dim=1)", language=Language.Fortran)
    assert parsed.tostring(language=Language.C) == "size(a, 1)"
    assert "dim" in parsed.tostring(language=Language.Fortran)


@pytest.mark.slow
class TestSizeDimKeyword(util.F2PyTest):
    # `size(a, dim=1)` in a result shape must not mention `dim` in C.
    suffix = ".f90"
    code = """
function firstcol(a) result(y)
  real, intent(in) :: a(:,:)
  real :: y(size(a, dim=1))
  y = a(:, 1)
end function
"""

    def test_leading_extent(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = self.module.firstcol(np.asfortranarray(a))
        assert len(out) == 2
        assert out[0] == np.float32(1.0)
