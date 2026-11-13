import pytest

from . import util


@pytest.mark.slow
class TestOptionalDerived(util.F2PyTest):
    # `err` is an optional derived type of intrinsic components.
    # It is left absent so `add` still exports. A required derived
    # argument does not take the whole module down.
    suffix = ".f90"
    code = """
module m
  type :: state
    integer :: flag = 0
    character(len=8) :: message = ""
  end type state
contains
  function add(x, err) result(y)
    integer, intent(in) :: x
    type(state), intent(out), optional :: err
    integer :: y
    y = x + 1
    if (present(err)) err%flag = 7
  end function add
  function inc(x) result(y)
    integer, intent(in) :: x
    integer :: y
    y = x + 2
  end function inc
  function need(x, err) result(y)
    integer, intent(in) :: x
    type(state), intent(out) :: err
    integer :: y
    err%flag = x
    y = x
  end function need
end module m
"""

    def test_optional_derived_is_absent(self):
        assert self.module.m.add(3) == 4
        assert self.module.m.inc(3) == 5
        assert not hasattr(self.module.m, 'need')
