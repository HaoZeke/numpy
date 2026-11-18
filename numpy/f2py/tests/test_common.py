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
class TestCommonMultiple(util.F2PyTest):
    suffix = ".f90"
    code = """
        subroutine init_named()
          integer :: ia, ib, ic
          common /blk1/ ia, ib /blk2/ ic
          ia = 4
          ib = 5
          ic = 6
        end subroutine

        subroutine init_blank()
          integer, parameter :: n = 4
          integer :: id, arr, ie
          common id, arr(n/2) /blk3/ ie
          id = 7
          arr = [8, 9]
          ie = 10
        end subroutine
    """

    def test_multiple_blocks(self):
        self.module.init_named()
        self.module.init_blank()
        assert self.module.blk1.ia == 4
        assert self.module.blk1.ib == 5
        assert self.module.blk2.ic == 6
        assert self.module._BLNK_.id == 7
        np.testing.assert_array_equal(self.module._BLNK_.arr, [8, 9])
        assert self.module.blk3.ie == 10
