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
@pytest.mark.parametrize("code, block, var, setter, updated", [
    (
        """
module mcmod
  implicit none
  integer :: m
  common /mc/ m
end module
subroutine setm(v)
  use mcmod
  implicit none
  integer :: v
  m = v
end subroutine
""",
        "mc", "m", "setm", 4,
    ),
    (
        """
module typedefs
  implicit none
  integer, parameter :: rk = kind(1.0)
end module
module values
  use typedefs, only: rk
  implicit none
  real(kind=rk) :: x
  common /buf/ x
end module
subroutine setx(v)
  use values
  implicit none
  real(kind=rk) :: v
  x = v
end subroutine
""",
        "buf", "x", "setx", np.float32(2.5),
    ),
])
def test_common_from_used_module(code, block, var, setter, updated):
    mod = util.build_code(code, suffix=".f90")
    getattr(mod, setter)(updated)
    got = np.array(getattr(getattr(mod, block), var)).item()
    assert got == np.array(updated).item()
