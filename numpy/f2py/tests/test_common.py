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
class TestSlashInitCommon(util.F2PyTest):
    suffix = ".f"
    code = (
        "      subroutine s\n"
        "      integer a(3) /1, 2, 3/, n /9/\n"
        "      common /blk/ a, n\n"
        "      end\n"
    )

    def test_slash_init_keeps_dimension(self):
        np.testing.assert_array_equal(self.module.blk.a, [1, 2, 3])
        assert int(self.module.blk.n) == 9
