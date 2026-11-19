import pytest

from . import util


@pytest.mark.slow
class TestModuleEntry(util.F2PyTest):
    suffix = ".f90"
    code = """
module me
contains
  subroutine inc(a, b)
    integer, intent(in) :: a
    integer, intent(out) :: b
    b = a + 1
    return
    entry inc2(a, b)
    b = a + 2
  end subroutine

  function twice(a)
    integer :: a, twice, thrice
    twice = 2 * a
    return
    entry thrice(a)
    thrice = 3 * a
  end function
end module

module private_me
  private
  public :: inc_public
contains
  subroutine inc_private(a, b)
    integer, intent(in) :: a
    integer, intent(out) :: b
    b = a + 1
    return
    entry inc_public(a, b)
    b = a + 2
  end subroutine
end module
"""

    @pytest.mark.parametrize("name,expected", [
        ("inc2", 3),
        ("thrice", 3),
    ])
    def test_entry(self, name, expected):
        assert getattr(self.module.me, name)(1) == expected

    def test_host_and_public_entry(self):
        assert self.module.me.inc(1) == 2
        assert self.module.me.twice(1) == 2
        assert self.module.private_me.inc_public(1) == 3
