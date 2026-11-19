import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestKindPrefix(util.F2PyTest):
    suffix = ".f90"
    code = """
real(kind=kind(1.0d0)) function twice(x)
  real(kind=kind(1.0d0)), intent(in) :: x
  twice = x + x
end function

integer(kind=kind(1.0d0)) function big(n)
  integer, intent(in) :: n
  big = int(2, kind=kind(1.0d0)) ** n
end function
"""

    @pytest.mark.parametrize("name, arg, expected", [
        ("twice", np.float64(1.25), 2.5),
        ("big", 40, 2 ** 40),
    ])
    def test_kind_literal_prefix(self, name, arg, expected):
        assert getattr(self.module, name)(arg) == expected
