import textwrap

import numpy as np
import pytest

from numpy.f2py import crackfortran

from . import util


def test_use_only_keeps_operator_specs(tmp_path):
    src = tmp_path / "useop.f90"
    src.write_text(textwrap.dedent("""
        module user
        contains
          subroutine apply(n, out)
            use ops, only: operator(.myplus.) => operator(.plus.), operator(==)
            use rel, only: assignment(=)
            use kinds, only: wp => dp, ip
            use more, only: operator(.plus.)
            integer, intent(in) :: n(:)
            integer, intent(out) :: out
            out = n(1)
          end subroutine
        end module
        """))
    crackfortran.reset_global_f2py_vars()
    mod = crackfortran.crackfortran([str(src)])
    pyf = crackfortran.crack2fortran(mod)
    assert "use ops, only: operator(.myplus.)=>operator(.plus.),operator(==)" in pyf
    assert "use rel, only: assignment(=)" in pyf
    assert "use kinds, only: wp=>dp,ip" in pyf
    assert "use more, only: operator(.plus.)" in pyf


@pytest.mark.slow
def test_renamed_operator_in_wrapper():
    code = textwrap.dedent("""
        module ops
          interface operator(.plus.)
            module procedure add
          end interface
        contains
          integer function add(x, y)
            integer, intent(in) :: x, y
            add = x + y
          end function
        end module
        module user
        contains
          subroutine apply(n, out)
            use ops, only: operator(.myplus.) => operator(.plus.)
            integer, intent(in) :: n(:)
            integer, intent(out) :: out
            out = n(1) .myplus. 1
          end subroutine
        end module
        """)
    mod = util.build_code(code, suffix=".f90")
    got = mod.user.apply(np.array([2], dtype=np.int32))
    assert int(got) == 3
