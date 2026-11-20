import pytest

from . import util


@pytest.mark.slow
class TestEntryDummyDeps(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine foo(n, a, s)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: s
  real(8), intent(in) :: b(n)
  s = sum(a)
  return
  entry bar_scalar(n, s)
  s = dble(n) * 2.0d0
  return
  entry bar_other(n, b, s)
  s = sum(b) * 3.0d0
end subroutine foo
"""

    @pytest.mark.parametrize("name,arg,expected", [
        ("bar_scalar", 5, 10.0),
        ("bar_other", [1.0, 2.0], 9.0),
    ])
    def test_entry_dummy_deps(self, name, arg, expected):
        assert getattr(self.module, name)(arg) == expected
