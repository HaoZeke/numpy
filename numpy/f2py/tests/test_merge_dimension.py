import numpy as np
import pytest

from numpy.f2py.symbolic import Expr, Language

from . import util


def test_merge_dimension_is_c_ternary():
    # `mask=1<dim` is a keyword. In C it must be a ternary, not
    # `merge(...)` and not `1 .lt. dim`.
    expr = Expr.parse(
        "merge(size(x, 1), size(x, 2), mask=1<dim)",
        language=Language.Fortran,
    )
    c = expr.tostring(language=Language.C)
    assert "merge" not in c
    assert ".lt." not in c
    assert c == "(1<dim?size(x, 1):size(x, 2))"


@pytest.mark.slow
class TestMergeDimension(util.F2PyTest):
    # The result length is `merge(size(x,1), size(x,2), mask=1<dim)`.
    suffix = ".f90"
    code = """
module m
contains
  function pick(x, dim) result(y)
    real, intent(in) :: x(:,:)
    integer, intent(in) :: dim
    real :: y(merge(size(x, 1), size(x, 2), mask=1<dim))
    y = 1
  end function pick
end module m
"""

    def test_result_length_follows_merge(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        assert self.module.m.pick(x, 2).shape == (2,)
        assert self.module.m.pick(x, 1).shape == (3,)
