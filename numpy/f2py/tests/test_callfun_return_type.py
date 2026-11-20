import pytest

from . import util


@pytest.mark.slow
class TestCallfunReturnType(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine call_real(f, x, k)
  external f
  real, intent(in) :: x
  integer, intent(out) :: k
  k = f(x)
end subroutine call_real

subroutine call_int(ifn, n, y)
  external ifn
  integer, intent(in) :: n
  real, intent(out) :: y
  y = 2.0 * ifn(n)
end subroutine call_int
"""

    @pytest.mark.parametrize("name,cb,arg,expected", [
        ("call_real", lambda x: x * 2.5, 2.0, 5),
        ("call_int", lambda n: n + 1, 3, 8.0),
    ])
    def test_external_return_type(self, name, cb, arg, expected):
        assert getattr(self.module, name)(cb, arg) == expected
