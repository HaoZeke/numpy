import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCommonBlock(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "block.f")]

    def test_common_block(self):
        self.module.initcb()
        assert self.module.block.long_bn == np.array(1.0, dtype=np.float64)
        assert self.module.block.string_bn == np.array("2", dtype="|S1")
        assert self.module.block.ok == np.array(3, dtype=np.int32)


@pytest.mark.slow
class TestCommonWithUse(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "gh19161.f90")]

    def test_common_gh19161(self):
        assert self.module.data.x == 0


@pytest.mark.slow
@pytest.mark.parametrize("code,suffix,size", [
    (
        """
      subroutine small
      integer a(1)
      common /blk/ a
      end
      subroutine big
      integer a(4)
      common /blk/ a
      a(1) = 1
      a(2) = 2
      a(3) = 3
      a(4) = 4
      end
""",
        ".f",
        4,
    ),
    (
        """
subroutine small
  implicit none
  integer :: a(2)
  common /blk/ a
end subroutine
subroutine big
  implicit none
  integer, parameter :: n = 5
  integer :: a(n)
  integer :: i
  common /blk/ a
  do i = 1, n
    a(i) = i
  end do
end subroutine
""",
        ".f90",
        5,
    ),
])
def test_common_wider_declaration(code, suffix, size):
    module = util.build_code(code, suffix=suffix)
    assert module.blk.a.shape == (size,)
    module.big()
    np.testing.assert_array_equal(
        module.blk.a, np.arange(1, size + 1, dtype=module.blk.a.dtype))
