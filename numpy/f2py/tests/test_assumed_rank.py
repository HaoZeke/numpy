import pytest

from . import util


@pytest.mark.slow
class TestAssumedRank(util.F2PyTest):
    suffix = ".f90"
    code = """
module rankmod
contains
  subroutine unsupported(a)
    real, intent(in) :: a(..)
  end subroutine unsupported

  subroutine add_one(x, y)
    integer, intent(in) :: x
    integer, intent(out) :: y
    y = x + 1
  end subroutine add_one
end module rankmod
"""

    def test_sibling_still_wraps(self):
        assert self.module.rankmod.add_one(3) == 4
        assert not hasattr(self.module.rankmod, "unsupported")
