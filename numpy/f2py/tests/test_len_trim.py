import pytest

from . import util


@pytest.mark.slow
class TestLenTrim(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine trimmed(c, a)
        character(len=*), intent(in) :: c
        integer, intent(out) :: a(len_trim(c))
        a = len_trim(c)
      end subroutine trimmed

      subroutine nested(c, a)
        character(len=*), intent(in) :: c
        integer, intent(out) :: a(len(trim(c)))
        a = len(trim(c))
      end subroutine nested
    """

    def test_len_trim_dimension(self):
        assert self.module.trimmed("abc  ").shape == (3,)
        assert self.module.trimmed("   ").shape == (0,)

    def test_len_of_trim_dimension(self):
        assert self.module.nested("abc  ").shape == (3,)
