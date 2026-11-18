import pytest

from . import util


@pytest.mark.slow
class TestCharLenDepend(util.F2PyTest):
    # A character length that names another argument has to be applied
    # after that argument is known. Declaration order is not enough:
    # the character entity may be declared first.
    suffix = ".f90"
    code = """
character(len=n) function prefix_fun(n)
  implicit none
  integer, intent(in) :: n
  prefix_fun = 'ABC'
end function prefix_fun

subroutine decl_char_first(buf, n, out)
  character(len=n), intent(in) :: buf
  integer, intent(in) :: n
  integer, intent(out) :: out
  out = n + ichar(buf(1:1))
end subroutine decl_char_first
"""

    def test_length_follows_argument(self):
        assert self.module.prefix_fun(3) == b"ABC"
        assert self.module.prefix_fun(1) == b"A"
        assert self.module.decl_char_first(b"ab", 2) == 2 + ord("a")
