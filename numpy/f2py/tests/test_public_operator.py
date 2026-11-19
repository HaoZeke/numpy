import textwrap

import pytest

from numpy.f2py import crackfortran

from . import util


def test_public_operator_name_no_implicit_type(tmp_path):
    source = tmp_path / "operator.f90"
    source.write_text("module a\npublic :: operator(+)\nend module\n")
    module = crackfortran.crackfortran([str(source)])[0]
    assert "typespec" not in module["vars"]["operator(+)"]


@pytest.mark.slow
@pytest.mark.parametrize("op,calc,expect", [
    ("plus", "x + y", 3),
    ("times", "x * y", 6),
])
def test_public_renamed_operator(op, calc, expect):
    code = textwrap.dedent(f"""
        module a
          interface operator(.{op}.)
            module procedure combine
          end interface
        contains
          integer function combine(x, y)
            integer, intent(in) :: x, y
            combine = {calc}
          end function
        end module
        module b
          use a, only: operator(.my{op}.) => operator(.{op}.)
          public :: operator(.my{op}.)
        contains
          integer function apply(n)
            integer, intent(in) :: n
            apply = n .my{op}. {3 if op == 'times' else 1}
          end function
        end module
        """)
    mod = util.build_code(code, suffix=".f90")
    assert mod.b.apply(2) == expect
