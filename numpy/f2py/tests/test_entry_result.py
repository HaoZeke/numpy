import pytest

from . import util


class TestEntryResult(util.F2PyTest):
    suffix = ".f90"
    code = """
real function foo(n)
  integer, intent(in) :: n
  foo = n
  return
entry alt(n) result(iresult)
  iresult = n + 1
end function foo
"""

    @pytest.mark.parametrize("n", [3])
    def test_entry_result(self, n):
        got = self.module.alt(n)
        assert got == n + 1
        assert type(got) is int
