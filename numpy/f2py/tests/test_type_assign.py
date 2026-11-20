import pytest

from . import util


@pytest.mark.slow
class TestTypeAssign(util.F2PyTest):
    suffix = ".f90"
    code = """
        subroutine settype(n, type)
          integer, intent(in) :: n
          integer, intent(out) :: type
          type = n
        end subroutine settype

        function addtype(n) result(m)
          integer, intent(in) :: n
          integer :: m, type
          type = n
          m = type
        end function addtype
        """

    @pytest.mark.parametrize("routine, n", [("settype", 4), ("addtype", 5)])
    def test_assignment(self, routine, n):
        assert getattr(self.module, routine)(n) == n
