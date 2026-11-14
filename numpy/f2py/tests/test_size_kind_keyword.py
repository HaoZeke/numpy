import numpy as np
import pytest

from numpy.f2py.symbolic import Expr, Language

from . import util


def test_size_kind_keyword_is_omitted_in_c():
    parsed = Expr.parse("size(a, 1, kind=4)", language=Language.Fortran)
    assert parsed.tostring(language=Language.C) == "size(a, 1)"
    assert "kind" in parsed.tostring(language=Language.Fortran)


@pytest.mark.slow
class TestSizeKindKeyword(util.F2PyTest):
    # `size(a, 1, kind=4)` in a result shape must not mention `kind` in C.
    suffix = ".f90"
    code = """
function row(a) result(y)
  real, intent(in) :: a(:,:)
  real :: y(size(a, 1, kind=4))
  y = a(:, 1)
end function
"""

    def test_leading_extent(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = self.module.row(np.asfortranarray(a))
        assert len(out) == 2
        assert out[0] == np.float32(1.0)
