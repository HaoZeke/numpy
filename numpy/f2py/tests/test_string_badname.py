import pytest

from . import util


@pytest.mark.slow
class TestStringBadname(util.F2PyTest):
    # The dummy is named `string`, which f2py renames to `string_bn`.
    # `len(string)` must use that C name or the wrapper does not compile.
    suffix = ".f90"
    code = """
module m
contains
  function up(string) result(s)
    character(len=*), intent(in) :: string
    character(len=len(string)) :: s
    integer :: i
    do i = 1, len(string)
      if (string(i:i) >= 'a' .and. string(i:i) <= 'z') then
        s(i:i) = achar(iachar(string(i:i)) - 32)
      else
        s(i:i) = string(i:i)
      end if
    end do
  end function up
end module m
"""

    def test_character_argument_named_string(self):
        assert self.module.m.up('abC') == b'ABC' or self.module.m.up('abC') == 'ABC'
