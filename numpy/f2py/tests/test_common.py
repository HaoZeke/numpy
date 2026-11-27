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
class TestCommonIntrinsicExtent(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine s
  implicit none
  integer :: a(mod(7, 4))
  integer :: b(abs(-3))
  integer :: d(modulo(-7, 4))
  common /blk/ a, b, d
  a = 6
  b = 4
  d = 8
end subroutine

subroutine t
  implicit none
  character(len=mod(7, 4)) :: c
  character*(abs(-3)) :: e
  common /ch/ c, e
  c = 'xyz'
  e = 'uvw'
end subroutine
"""

    def test_intrinsic_common_extent(self):
        self.module.s()
        self.module.t()
        np.testing.assert_array_equal(self.module.blk.a, [6, 6, 6])
        np.testing.assert_array_equal(self.module.blk.b, [4, 4, 4])
        np.testing.assert_array_equal(self.module.blk.d, [8])
        assert self.module.ch.c == b'xyz'
        assert self.module.ch.e == b'uvw'
