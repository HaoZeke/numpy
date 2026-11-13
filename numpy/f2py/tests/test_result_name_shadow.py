import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestResultNameShadow(util.F2PyTest):
    # `string` is renamed to string_bn. The result clause has to follow
    # that rename, or the character declaration is dropped and the result
    # stays an implicit real.
    suffix = ".f90"
    code = """
module m
contains
  function lower(str) result(string)
    character(*), intent(in) :: str
    character(len(str)) :: string
    string = str
  end function lower
end module m
"""

    def test_character_result_named_string(self):
        got = self.module.m.lower("AbC")
        assert got == b"AbC"
        assert self.module.m.lower("") == b""
