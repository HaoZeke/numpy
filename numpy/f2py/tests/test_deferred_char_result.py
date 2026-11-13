import pytest

from . import util


@pytest.mark.slow
class TestDeferredCharResult(util.F2PyTest):
    # The result length is not the input length. Fortran allocates it
    # and the wrapper copies that many characters.
    suffix = ".f90"
    code = """
module m
contains
  function f(s, n) result(t)
    character(*), intent(in) :: s
    integer, intent(in) :: n
    character(:), allocatable :: t
    if (n <= len(s)) then
      t = s(1:n)
    else
      t = s // repeat('x', n - len(s))
    end if
  end function
end module
"""

    def test_shorter_and_longer_than_input(self):
        assert self.module.m.f("abcd", 1) == b"a"
        assert self.module.m.f("ab", 5) == b"abxxx"
        assert self.module.m.f("ab", 2) == b"ab"
