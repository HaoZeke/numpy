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
@pytest.mark.parametrize("inner,expect", [
    (
        "integer :: c\n    common /b2/ c\n    c = 9",
        9,
    ),
    (
        "integer :: c(2)\n    common /b2/ c\n    c(1) = 2\n    c(2) = 3",
        [2, 3],
    ),
])
def test_common_in_contained_routine(inner, expect):
    code = f"""
module m
  implicit none
  integer :: a
  common /b1/ a
contains
  subroutine s
    implicit none
    {inner}
    a = 4
  end subroutine s
end module m
"""
    mod = util.build_code(code, suffix=".f90")
    mod.m.s()
    assert int(mod.b1.a) == 4
    got = np.array(mod.b2.c)
    if isinstance(expect, list):
        np.testing.assert_array_equal(got, expect)
    else:
        assert int(got) == expect
