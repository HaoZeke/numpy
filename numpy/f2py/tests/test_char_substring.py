import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("expr,size", [
    ("'1.2'(2:)", "2"),
    ("word(:2)", "2"),
])
def test_char_substring_parameter_length(tmp_path, expr, size):
    source = tmp_path / "sub.f90"
    source.write_text(textwrap.dedent(f"""\
        subroutine s(x)
          implicit none
          character(len=*), parameter :: word = 'abcdef'
          integer, parameter :: n = len({expr})
          real, intent(out) :: x(n)
          x = 1.0
        end subroutine s
        """))
    mod = crackfortran.crackfortran([str(source)])
    assert mod[0]["vars"]["x"]["dimension"] == [size]
