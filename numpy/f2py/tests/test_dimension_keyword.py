import pytest

from . import util


@pytest.mark.slow
class TestDimensionKeyword(util.F2PyTest):
    # `mask=1<dim` is a keyword whose value is a comparison.
    # It used to be parsed as `(mask=1) < dim` and the cracker aborted.
    suffix = ".f90"
    code = """
module m
contains
  function nrm(a, dim) result(y)
    real, intent(in) :: a(:,:)
    integer, intent(in) :: dim
    real :: y
    real :: tmp(merge(size(a, 1), size(a, 2), mask=1<dim))
    tmp = 0
    y = a(1, 1) + tmp(1)
  end function nrm
end module m
"""

    def test_keyword_comparison_dimension(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        assert self.module.m.nrm(a, 1) == 1
        assert self.module.m.nrm(a, 0) == 1
