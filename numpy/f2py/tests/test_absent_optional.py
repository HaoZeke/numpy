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
  function addn_default(x, n) result(y)
    integer, intent(in) :: x
    integer, intent(in), optional :: n
    !f2py integer, intent(in) :: n = 7
    integer :: y
    y = x + n
  end function addn_default
  function scale(x, factor) result(y)
    real, intent(in) :: x
    real, intent(in), optional :: factor
    real :: y
    if (present(factor)) then
      y = x * factor
    else
      y = -1
    end if
  end function scale
  function scale_default(x, factor) result(y)
    real, intent(in) :: x
    real, intent(in), optional :: factor
    !f2py real, intent(in) :: factor = 2.0
    real :: y
    y = x * factor
  end function scale_default
  function addvec(a, b) result(y)
    real, intent(in) :: a(2)
    real, intent(in), optional :: b(2)
    real :: y
    y = sum(a)
    if (present(b)) y = y + sum(b)
  end function addvec
end module optmod
"""

    def test_omitted_logical_is_absent(self):
        assert self.module.optmod.pick(3.0) == 0
        assert self.module.optmod.pick(3.0, True) == 3
        assert self.module.optmod.pick(3.0, False) == -3

    def test_omitted_integer_is_absent(self):
        assert self.module.optmod.addn(4) == -1
        assert self.module.optmod.addn(4, 2) == 6

    def test_omitted_real_is_absent(self):
        assert self.module.optmod.scale(3.0) == -1
        assert self.module.optmod.scale(3.0, 4.0) == 12

    def test_f2py_initializer_is_still_present(self):
        # An explicit `!f2py` value is the default, not an absent argument.
        assert self.module.optmod.addn_default(4) == 11
        assert self.module.optmod.addn_default(4, 2) == 6
        assert self.module.optmod.scale_default(3.0) == 6
        assert self.module.optmod.scale_default(3.0, 4.0) == 12

    def test_explicit_shape_array_is_unchanged(self):
        # An omitted explicit-shape array is still a present zero array.
        assert self.module.optmod.addvec([1.0, 2.0]) == 3
        assert self.module.optmod.addvec([1.0, 2.0], [10.0, 20.0]) == 33
