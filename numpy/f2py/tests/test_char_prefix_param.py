import pytest

from . import util


@pytest.mark.slow
class TestCharPrefixParam(util.F2PyTest):
    # The length is a parameter of the host module, named in the
    # function prefix. The wrapper must use that length.
    suffix = ".f90"
    code = """
module m
  integer, parameter :: nlen = 6
contains
  character(len=nlen) function foo()
    foo = 'abcdef'
  end function foo
  character*(nlen) function bar()
    bar = 'ghijkl'
  end function bar
end module m
"""

    @pytest.mark.parametrize(
        "name,expect", [("foo", b"abcdef"), ("bar", b"ghijkl")]
    )
    def test_param_length(self, name, expect):
        assert getattr(self.module.m, name)() == expect
