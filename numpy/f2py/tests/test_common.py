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
@pytest.mark.parametrize("binding", [
    "bind(c) :: /blk/",
    'bind(c, name="MyBlk") :: /blk/',
])
def test_bindc_common(binding):
    code = f"""
subroutine setx()
  use iso_c_binding
  implicit none
  integer(c_int) :: x
  common /blk/ x
  {binding}
  x = 42
end subroutine
"""
    mod = util.build_code(code, suffix=".f90")
    mod.setx()
    assert int(np.array(mod.blk.x)) == 42
