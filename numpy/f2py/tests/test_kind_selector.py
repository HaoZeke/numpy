import pytest

from . import util


@pytest.mark.slow
class TestKindSelector(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
  implicit none
  real(kind=selected_real_kind(15, 307)) :: x
  integer(kind=selected_int_kind(18)) :: n
contains
  subroutine setvals()
    x = 1.25d0
    n = 4294967297_8
  end subroutine setvals
end module m
"""

    @pytest.mark.parametrize("name,value", [
        ("x", 1.25),
        ("n", 4294967297),
    ])
    def test_inline_kind_selector(self, name, value):
        self.module.m.setvals()
        got = getattr(self.module.m, name)
        assert got.item() == value
