import textwrap

import pytest

from numpy.f2py import crackfortran


@pytest.mark.parametrize("header, source", [
    ("type, bind(c) :: vec", """
        module m
          type, bind(c) :: vec
            integer :: n
            real :: data(4)
          end type vec
        end module m
    """),
    ("type, extends(base) :: child", """
        module m
          type :: base
            integer, allocatable :: vals(:)
          end type base
          type, extends(base) :: child
            integer :: tag(2)
          end type child
        end module m
    """),
])
def test_type_header(tmp_path, header, source):
    path = tmp_path / "t.f90"
    path.write_text(textwrap.dedent(source))
    pyf = crackfortran.crack2fortran(crackfortran.crackfortran([str(path)]))
    assert header in pyf
