import textwrap

import pytest

from numpy.f2py import crackfortran

from . import util


def _crack(src, tmp_path):
    path = tmp_path / "literal_case.f90"
    path.write_text(textwrap.dedent(src).lstrip("\n"), encoding="ascii")
    return crackfortran.crackfortran([str(path)])[0]


class TestCharacterLiteralCase:
    def test_quotes_keep_case_while_names_are_lowered(self, tmp_path):
        mod = _crack(
            """
            SUBROUTINE CopyName(A, B)
              CHARACTER(LEN=*), OPTIONAL :: A = 'A!b'
              CHARACTER(LEN=*), OPTIONAL :: B = 'It''s OK'
            END SUBROUTINE CopyName
            """,
            tmp_path,
        )
        assert mod["name"] == "copyname"
        assert list(mod["vars"]) == ["a", "b"]
        assert mod["vars"]["a"]["="] == "'A!b'"
        assert mod["vars"]["b"]["="] == "'It''s OK'"


@pytest.mark.slow
class TestCharacterLiteralDefault(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent(
        """
        subroutine copyname(out, name)
          character(len=5), intent(out) :: out
          character(len=5), intent(in) :: name
          !f2py character, optional :: name = 'Hello'
          out = name
        end subroutine copyname
        """
    )

    def test_omitted_default_keeps_case(self):
        assert self.module.copyname() == b"Hello"
