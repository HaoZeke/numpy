import pytest

from . import util


@pytest.mark.slow
class TestNaryMinMax(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine cap(a, n, m)
        integer, intent(in) :: n, m
        real, intent(out) :: a(max(n, m, 1))
      end subroutine cap

      subroutine floor(a, n, m)
        integer, intent(in) :: n, m
        real, intent(out) :: a(min(n, m, 4))
      end subroutine floor

      subroutine nested(a, n, m)
        integer, intent(in) :: n, m
        real, intent(out) :: a(2 * max(n, min(m, 3, 4), 1))
      end subroutine nested

      subroutine checked(a, n, m)
        integer, intent(in) :: n, m
!f2py check(m == min(m, 4, 5)) m
        real, intent(out) :: a(max(n, m, 1))
      end subroutine checked

      subroutine dependent(a, n, m, k)
        integer, intent(in) :: n, m
!f2py integer intent(hide), depend(n, m) :: k = max(n, m, 1)
        real, intent(out) :: a(k)
      end subroutine dependent
    """

    def test_nary_min_max(self):
        assert self.module.cap(2, 5).shape == (5,)
        assert self.module.floor(8, 6).shape == (4,)
        assert self.module.nested(2, 5).shape == (6,)
        assert self.module.checked(5, 2).shape == (5,)
        assert self.module.dependent(2, 5).shape == (5,)
        with pytest.raises(Exception):
            self.module.checked(2, 5)
