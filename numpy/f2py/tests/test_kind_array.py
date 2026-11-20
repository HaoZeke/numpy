import pytest

from . import util

REAL_SRC = """
function foo()
  integer, parameter :: ks(2) = (/4, 8/)
  real(kind=ks(2)) :: foo
  foo = 16777217.0d0
end function
"""

COMPLEX_SRC = """
function foo()
  integer, parameter :: ks(2) = (/4, 8/)
  complex(kind=ks(2)) :: foo
  foo = (16777217.0d0, -0.5d0)
end function
"""

EXPRESSION_SRC = """
function foo()
  integer, parameter :: ks(2) = (/4, 8/)
  real(kind=ks(1 + 1)) :: foo
  foo = 16777217.0d0
end function
"""

NESTED_EXPRESSION_SRC = """
function foo()
  integer, parameter :: ks(2) = (/4, 8/)
  integer, parameter :: idx(2) = (/1, 2/)
  real(kind=ks(idx(1) + 1)) :: foo
  foo = 16777217.0d0
end function
"""

CHAR_LEN_SRC = """
subroutine foo(s)
  integer, parameter :: ls(2) = (/3, 5/)
  character(len=ls(1 + 1)), intent(out) :: s
  s = 'hello'
end subroutine
"""


@pytest.mark.slow
@pytest.mark.parametrize(("src", "expected"), [
    (REAL_SRC, 16777217.0),
    (COMPLEX_SRC, 16777217.0 - 0.5j),
    (EXPRESSION_SRC, 16777217.0),
    (NESTED_EXPRESSION_SRC, 16777217.0),
    (CHAR_LEN_SRC, b"hello"),
])
def test_kind_indexes_parameter_array(src, expected):
    mod = util.build_code(src, suffix=".f90")
    assert mod.foo() == expected
