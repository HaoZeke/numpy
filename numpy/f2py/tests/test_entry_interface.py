import pytest

from . import util


@pytest.mark.slow
class TestEntryInterface(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine total(a, b)
  real(8), intent(in) :: a(:)
  real(8), intent(out) :: b
  b = sum(a)
  return
  entry prod(a, b)
  b = product(a)
end subroutine

subroutine scale(a, k, b)
  real(8), intent(in) :: a(:)
  integer, intent(in) :: k
  real(8), intent(out) :: b
  b = k * sum(a)
  return
  entry scale2(a, b)
  b = 2 * sum(a)
end subroutine
"""

    @pytest.mark.parametrize("name,expected", [
        ("prod", 6.0),
        ("scale2", 12.0),
    ])
    def test_entry(self, name, expected):
        assert getattr(self.module, name)([1.0, 2.0, 3.0]) == expected
