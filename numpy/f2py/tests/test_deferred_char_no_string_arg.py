import pytest

from . import util


@pytest.mark.slow
class TestDeferredCharNoStringArg(util.F2PyTest):
    # The only dummy is an integer, so nothing else asks for the
    # `string` typedef. The result declaration has to be a real C type.
    suffix = ".f90"
    code = """
module m
contains
  function xs(n) result(t)
    integer, intent(in) :: n
    character(len=:), allocatable :: t
    t = repeat('x', n)
  end function
end module
"""

    def test_allocated_length_with_no_character_argument(self):
        assert self.module.m.xs(3) == b"xxx"
        assert self.module.m.xs(1) == b"x"
