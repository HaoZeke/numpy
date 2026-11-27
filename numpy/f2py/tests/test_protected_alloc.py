import numpy as np
import pytest
from numpy.testing import assert_array_equal

from . import util


@pytest.mark.slow
class TestProtectedAllocatable(util.F2PyTest):
    suffix = ".f90"
    code = """
module pm
  implicit none
  integer, protected :: n = 3
  real, protected, allocatable :: a(:,:)
  real, protected, allocatable :: c(:,:,:)
  real, protected, allocatable :: z(:)
  real, protected, allocatable :: zz(:,:)
  real, protected, allocatable :: zrow(:,:)
  real, allocatable :: free(:,:)
contains
  subroutine fill()
    if (.not. allocated(a)) allocate(a(2, 3))
    a = reshape((/1., 2., 3., 4., 5., 6./), (/2, 3/))
    if (.not. allocated(c)) allocate(c(2, 2, 2))
    c = 1.0
  end subroutine fill

  subroutine fill0()
    if (.not. allocated(z)) allocate(z(0))
    if (.not. allocated(zz)) allocate(zz(0, 0))
    if (.not. allocated(zrow)) allocate(zrow(0, 5))
  end subroutine fill0

  real function asum()
    if (.not. allocated(a)) then
      asum = -1
    else
      asum = sum(a)
    end if
  end function asum
end module pm
"""

    def test_protected_rank2_and_rank3(self):
        m = self.module.pm
        assert m.a is None
        assert m.asum() == -1
        with pytest.raises(ValueError):
            m.a = np.ones((2, 3), dtype=np.float32)
        assert m.a is None

        m.fill()
        expected = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32)
        assert m.a.shape == (2, 3)
        assert_array_equal(m.a, expected)
        assert m.asum() == 21
        assert m.c.shape == (2, 2, 2)
        assert m.c.sum() == 8

        updated = np.array([[9, 8, 7], [6, 5, 4]], dtype=np.float32)
        m.a = updated
        assert_array_equal(m.a, updated)
        assert m.asum() == 39
        with pytest.raises(ValueError):
            m.a = np.ones((4, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            m.a = None
        assert_array_equal(m.a, updated)
        assert m.asum() == 39

        cubed = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        m.c = cubed
        assert_array_equal(m.c, cubed)
        with pytest.raises(ValueError):
            m.c = np.ones((2, 2, 3), dtype=np.float32)

        assert m.n == 3
        m.n = 4
        assert m.n == 4
        m.free = np.ones((2, 2), dtype=np.float32)
        m.free = np.arange(3, dtype=np.float32).reshape(3, 1)
        assert m.free.shape == (3, 1)
        assert m.free.sum() == 3
        m.free = None
        assert m.free is None

    def test_protected_zero_extent_none(self):
        m = self.module.pm
        m.fill0()
        assert m.z.shape == (0,)
        assert m.zz.shape == (0, 0)
        assert m.zrow.shape == (0, 5)

        m.z = np.zeros((0,), dtype=np.float32)
        assert m.z.shape == (0,)
        m.zz = np.zeros((0, 0), dtype=np.float32)
        assert m.zz.shape == (0, 0)
        m.zrow = np.zeros((0, 5), dtype=np.float32)
        assert m.zrow.shape == (0, 5)

        with pytest.raises(ValueError):
            m.z = None
        assert m.z.shape == (0,)
        with pytest.raises(ValueError):
            m.zz = None
        assert m.zz.shape == (0, 0)
        with pytest.raises(ValueError):
            m.zrow = None
        assert m.zrow.shape == (0, 5)
