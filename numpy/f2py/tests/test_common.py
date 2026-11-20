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
@pytest.mark.parametrize("decl", ["real(8) :: x(3)",
                                  "use kinds\n  real(rk) :: x(3)"])
def test_common_skips_unrelated_use(decl):
    code = f"""
module kinds
  integer, parameter :: rk = kind(1.0d0)
end module kinds
module state
  real(8) :: x = 1d0
end module state
subroutine setc()
  {decl}
  common /c/ x
  x = 2d0
end subroutine setc
subroutine getx(y)
  use state
  real(8), intent(out) :: y
  y = x
end subroutine getx
"""
    mod = util.build_code(code, suffix=".f90")
    mod.setc()
    assert np.array_equal(mod.c.x, [2.0, 2.0, 2.0])
