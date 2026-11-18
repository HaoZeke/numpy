import pytest

from . import util


@pytest.mark.slow
class TestEnumBound(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine fill(a)
  enum, bind(c)
    enumerator :: negative = -2
    enumerator :: bound = negative + 7, next_bound
  end enum
  integer, intent(out) :: a(next_bound)
  a = negative
end subroutine fill
"""

    def test_enumerator_bound(self):
        a = self.module.fill()
        assert a.shape == (6,)
        assert list(a) == [-2] * 6
