import pytest

from . import util


@pytest.mark.slow
class TestSubmoduleSingleModuleDef(util.F2PyTest):
    # Each submodule is the parent module. One f2py_<parent>_def and one
    # f2pyinit<parent> have to cover every submodule, or the C file
    # redefines those symbols.
    suffix = ".f90"
    code = """
module parent
  interface
    module subroutine one(n)
      integer, intent(out) :: n
    end subroutine one
    module subroutine two(n)
      integer, intent(out) :: n
    end subroutine two
  end interface
end module parent

submodule (parent) child1
contains
  module subroutine one(n)
    integer, intent(out) :: n
    n = 1
  end subroutine one
end submodule child1

submodule (parent) child2
contains
  module subroutine two(n)
    integer, intent(out) :: n
    n = 2
  end subroutine two
end submodule child2
"""

    def test_both_submodules_are_one_module(self):
        assert self.module.parent.one() == 1
        assert self.module.parent.two() == 2
