import pytest

from . import util


@pytest.mark.slow
class TestEmptyModuleHook(util.F2PyTest):
    # A module whose only contents were skipped still gets a hook.
    # That hook used to make the extension fail to import.
    suffix = ".f90"
    code = """
module norms
  type :: state
    integer :: code
  end type
end module
module has
contains
  integer function one()
    one = 1
  end function
end module
"""

    def test_sibling_routine_imports(self):
        assert self.module.has.one() == 1
