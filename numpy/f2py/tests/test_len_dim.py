import pytest

from . import util


class TestLenDimension(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine fill(c, a)
  character(len=*), intent(in) :: c
  integer, intent(out) :: a(len(c))
  integer :: i
  do i = 1, len(c)
    a(i) = i
  end do
end subroutine fill
"""

    @pytest.mark.parametrize("text", ["ab", "wxyz"])
    def test_len_extent(self, text):
        got = self.module.fill(text)
        assert list(got) == list(range(1, len(text) + 1))
