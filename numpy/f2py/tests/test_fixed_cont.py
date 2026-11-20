import pytest

import numpy as np
from numpy.f2py import crackfortran

from . import util


class TestFixedCont(util.F2PyTest):
    suffix = ".f"
    code = """
      subroutine fillten(a)
      integer a(1
     +0)
      integer i
      do i = 1, 10
        a(i) = i
      end do
      end

      subroutine fillpar(a)
      integer n
      parameter (n = 1
     +0)
      integer a(n)
      integer i
      do i = 1, n
        a(i) = n
      end do
      end
"""

    @pytest.mark.parametrize("routine,expected", [
        ("fillten", list(range(1, 11))),
        ("fillpar", [10] * 10),
    ])
    def test_split_token(self, routine, expected):
        a = np.zeros(10, dtype=np.int32)
        getattr(self.module, routine)(a)
        assert a.tolist() == expected


class TestFixedContWords:
    def crack(self, tmp_path, lines):
        fpath = tmp_path / "cont.f"
        fpath.write_text("\n".join(lines) + "\n")
        return crackfortran.crackfortran([str(fpath)])

    def test_word_boundary_kept(self, tmp_path):
        mod = self.crack(tmp_path, [
            "      subroutine s(a)",
            "      integer",
            "     +a(3)",
            "      end",
            "",
            "      integer function",
            "     +f(x)",
            "      integer x",
            "      f = x",
            "      end",
        ])
        assert [b["name"] for b in mod] == ["s", "f"]
        var = mod[0]["vars"]["a"]
        assert var["typespec"] == "integer"
        assert var["dimension"] == ["3"]
        assert mod[1]["vars"]["f"]["typespec"] == "integer"

    def test_split_digits(self, tmp_path):
        mod = self.crack(tmp_path, [
            "      subroutine s(a)",
            "      integer a(1",
            "     +0, 2",
            "C     a comment between the parts",
            "     +5)",
            "      end",
        ])
        assert mod[0]["vars"]["a"]["dimension"] == ["10", "25"]

    def test_string_kept(self, tmp_path):
        mod = self.crack(tmp_path, [
            "      subroutine s(a)",
            "      character*10 a",
            "cf2py character*10 optional :: a = 'ab1",
            "cf2py+2cd'",
            "      end",
        ])
        assert mod[0]["vars"]["a"]["="] == "'ab1 2cd'"
