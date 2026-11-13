import pytest

from . import util


@pytest.mark.slow
class TestAbsentOptional(util.F2PyTest):
    # An omitted optional used to arrive as a present .false. / 0.
    suffix = ".f90"
    code = """
module optmod
contains
  function pick(x, flag) result(y)
    real, intent(in) :: x
    logical, intent(in), optional :: flag
    real :: y
    if (present(flag)) then
      if (flag) then
        y = x
      else
        y = -x
      end if
    else
      y = 0
    end if
  end function pick
  function addn(x, n) result(y)
    integer, intent(in) :: x
    integer, intent(in), optional :: n
    integer :: y
    if (present(n)) then
      y = x + n
    else
      y = -1
    end if
  end function addn
end module optmod
"""

    def test_omitted_logical_is_absent(self):
        assert self.module.optmod.pick(3.0) == 0
        assert self.module.optmod.pick(3.0, True) == 3
        assert self.module.optmod.pick(3.0, False) == -3

    def test_omitted_integer_is_absent(self):
        assert self.module.optmod.addn(4) == -1
        assert self.module.optmod.addn(4, 2) == 6
