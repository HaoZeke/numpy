import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestShapeDefault(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine srank(a, n, out)
  integer, intent(in) :: n
  real, intent(in) :: a(4, 2)
  integer, intent(out) :: out
  !f2py integer optional, intent(in), depend(a) :: n = size(shape(a))
  out = n
end

subroutine srank1(a, n, out)
  integer, intent(in) :: n
  real, intent(in) :: a(n)
  integer, intent(out) :: out
  !f2py integer optional, intent(in), depend(a) :: n = shape(a, 0)
  out = n
end

module mrank
contains
  subroutine msrank(a, n, out)
    integer, intent(in) :: n
    real, intent(in) :: a(2, 3, 4)
    integer, intent(out) :: out
    !f2py integer optional, intent(in), depend(a) :: n = size(shape(a))
    out = n
  end subroutine msrank
end module mrank
"""

    def test_size_shape_default(self):
        a = np.zeros((4, 2), dtype=np.float32)
        assert self.module.srank(a) == 2
        assert self.module.srank(a, 5) == 5
        assert self.module.srank(n=7, a=a) == 7

    def test_shape_dim_default(self):
        a = np.zeros(4, dtype=np.float32)
        assert self.module.srank1(a) == 4
        assert self.module.srank1(a, 2) == 2

    def test_module_size_shape(self):
        a = np.zeros((2, 3, 4), dtype=np.float32)
        assert self.module.mrank.msrank(a) == 3
        assert self.module.mrank.msrank(a, 1) == 1
        assert self.module.mrank.msrank(n=8, a=a) == 8
