import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestMergeCheck(util.F2PyTest):
    # The consistency check for this dummy is emitted into C. A Fortran
    # `merge` or `.lt.` there does not compile.
    suffix = ".f90"
    code = """
subroutine saw(x, dim, center, n)
  real, intent(in) :: x(:,:)
  integer, intent(in) :: dim
  real, intent(in) :: center(merge(size(x, 1), size(x, 2), mask=1<dim))
  integer, intent(out) :: n
  n = size(center)
end subroutine saw
"""

    def test_check_compiles_and_reports_the_extent(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        center = np.zeros(2, dtype=np.float32)
        assert self.module.saw(x, 2, center) == 2
