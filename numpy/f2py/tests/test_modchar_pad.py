import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestModuleCharacterAssign(util.F2PyTest):
    suffix = ".f90"
    code = """
        module m
          character(len=4) :: b = 'abcd'
          character(len=4) :: a = 'WXYZ'
          character(len=3) :: words(2) = (/'opq', 'rst'/)
          character(len=2) :: g(2, 3)
          character(len=4), allocatable :: names(:)
        contains
          logical function b_is_ab()
            b_is_ab = b == 'ab'
          end function b_is_ab

          logical function a_is_wxyz()
            a_is_wxyz = a == 'WXYZ'
          end function a_is_wxyz

          logical function a_is_hi()
            a_is_hi = a == 'hi'
          end function a_is_hi

          logical function words_match()
            words_match = words(1) == 'A' .and. words(2) == 'BB'
          end function words_match

          logical function g_match()
            g_match = g(1, 1) == 'a' .and. g(2, 1) == 'd' .and. &
                      g(1, 2) == 'b' .and. g(2, 2) == 'e' .and. &
                      g(1, 3) == 'c' .and. g(2, 3) == 'f'
          end function g_match

          logical function name_match()
            name_match = allocated(names)
            if (name_match) then
              name_match = names(1) == 'ab' .and. names(2) == 'cdef' .and. &
                           names(3) == 'Z'
            end if
          end function name_match
        end module m
    """

    def test_assignment(self):
        m = self.module.m
        m.b = b"ab"
        assert m.b.dtype == np.dtype("S4")
        assert m.b.tobytes() == b"ab  "
        assert m.b_is_ab()
        m.b = b"abcdefgh"
        assert m.b.tobytes() == b"abcd"
        assert m.a.tobytes() == b"WXYZ"
        assert m.a_is_wxyz()
        m.a = "hi"
        assert m.a.tobytes() == b"hi  "
        assert m.a_is_hi()
        m.words = [b"A", b"BB"]
        assert m.words.dtype == np.dtype("S3")
        assert m.words.tobytes() == b"A  BB "
        assert m.words_match()
        m.words = ["x", "yz"]
        assert m.words.tobytes() == b"x  yz "
        m.g = [["a", "b", "c"], ["d", "e", "f"]]
        assert m.g.dtype == np.dtype("S2")
        assert m.g.tobytes() == b"a b c d e f "
        assert m.g_match()
        m.names = [b"ab", b"cdef", b"Z"]
        assert m.names.dtype == np.dtype("S4")
        assert m.names.tobytes() == b"ab  cdefZ   "
        assert m.name_match()
