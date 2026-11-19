import pytest

from . import util


@pytest.mark.slow
class TestCheckOtherArg(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine cmp(a, n)
  implicit none
  real :: a
  integer :: n
  !f2py real, intent(in), check(a > n) :: a
  !f2py integer, intent(in) :: n
  if (a > n) then
    return
  end if
end subroutine cmp
"""

    @pytest.mark.parametrize(("a", "n", "accepted"), [
        (-1.0, -5, True),
        (1.0, 5, False),
    ])
    def test_check_uses_other_argument(self, a, n, accepted):
        if accepted:
            self.module.cmp(a, n)
        else:
            with pytest.raises(Exception):
                self.module.cmp(a, n)
