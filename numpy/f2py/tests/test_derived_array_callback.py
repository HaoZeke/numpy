import pytest

from . import util


@pytest.mark.slow
class TestDerivedArrayCallback(util.F2PyTest):
    # An allocatable derived-type array has no C mapping. The callback
    # that receives it has to be rejected so the rest of the module builds.
    suffix = ".f90"
    code = """
module m
  type :: item
    integer :: n
  end type item
contains
  subroutine ok(n)
    integer, intent(out) :: n
    n = 3
  end subroutine ok
  subroutine run(collect)
    type(item), allocatable :: items(:)
    external collect
    call collect(items)
  end subroutine run
end module m
"""

    def test_module_builds_without_the_callback(self):
        assert self.module.m.ok() == 3
        assert not hasattr(self.module.m, "run")
