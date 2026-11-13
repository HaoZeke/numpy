import pytest

from . import util


@pytest.mark.slow
class TestReservedNameLength(util.F2PyTest):
    # `string` is renamed to `string_bn` in the wrapper. The length
    # expression has to use that same spelling, or the C file refers to
    # capi_string_len, which was never declared.
    suffix = ".f90"
    code = """
function f(string) result(res)
  character(*), intent(in) :: string
  character(len(string)) :: res
  res = string
end function
"""

    def test_character_dummy_named_string(self):
        assert self.module.f("AbC") == b"AbC"
        assert self.module.f("") == b""
