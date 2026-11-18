import textwrap

import pytest

from numpy.f2py import crackfortran

from . import util


def _crack(src, tmp_path):
    path = tmp_path / "logical_init.f90"
    path.write_text(textwrap.dedent(src).lstrip("\n"), encoding="ascii")
    return crackfortran.crackfortran([str(path)])[0]


class TestLogicalInitializer:
    def test_unary_initializers_and_spaced_kinds(self, tmp_path):
        mod = _crack(
            """
            subroutine s(flag, n)
              logical, parameter :: x = .false.
              logical ( kind = 4 ), optional :: flag = .not.x
              integer ( kind = 4 ), optional :: n = - 1
            end subroutine s
            """,
            tmp_path,
        )
        assert list(mod["vars"]) == ["flag", "n"]
        assert mod["vars"]["flag"]["kindselector"] == {"kind": "4"}
        assert mod["vars"]["flag"]["="] == "1"
        assert mod["vars"]["n"]["kindselector"] == {"kind": "4"}
        assert mod["vars"]["n"]["="] == "-1"


@pytest.mark.slow
class TestLogicalInitializerDefault(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent(
        """
        subroutine flag_not(out, flag)
          integer, intent(out) :: out
          logical, intent(in) :: flag
          !f2py logical, optional :: flag = .not. .false.
          if (flag) then
            out = 1
          else
            out = 0
          end if
        end subroutine flag_not

        subroutine flag_and(out, flag)
          integer, intent(out) :: out
          logical, intent(in) :: flag
          !f2py logical, optional :: flag = .true. .and. .false.
          if (flag) then
            out = 1
          else
            out = 0
          end if
        end subroutine flag_and
        """
    )

    def test_not_false_default(self):
        assert self.module.flag_not() == 1

    def test_and_default(self):
        assert self.module.flag_and() == 0
