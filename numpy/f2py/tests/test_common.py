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
@pytest.mark.parametrize("code,suffix,check", [
    (
        """
      subroutine s
      integer n
      parameter (n=4)
      character*(n) a
      integer b
      common /blk/ a, b
      a = 'wxyz'
      b = 9
      end
""",
        ".f",
        "scalar",
    ),
    (
        """
subroutine s
  implicit none
  integer, parameter :: n = 3
  character*(n) :: words(2)
  common /blk/ words
  words(1) = 'abc'
  words(2) = 'def'
end subroutine
""",
        ".f90",
        "array",
    ),
])
def test_common_parameter_character_length(code, suffix, check):
    module = util.build_code(code, suffix=suffix)
    module.s()
    if check == "scalar":
        assert module.blk.a == np.array("wxyz", dtype="|S4")
        assert module.blk.b == np.array(9, dtype=np.int32)
    else:
        assert module.blk.words.dtype == np.dtype("|S3")
        np.testing.assert_array_equal(module.blk.words, [b"abc", b"def"])
