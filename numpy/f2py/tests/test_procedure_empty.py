import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestProcedureEmpty(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
contains
  real function apply_r(f, x)
    procedure() :: f
    real :: x
    apply_r = f(x)
  end function apply_r
  integer function apply_i(f, n)
    procedure ( ) :: f
    integer :: f
    integer :: n
    apply_i = f(n)
  end function apply_i
end module m
"""

    @pytest.mark.parametrize("name,value,dtype", [
        ("apply_r", 1.5, np.float32),
        ("apply_i", 3, np.int32),
    ])
    def test_procedure_dummy(self, name, value, dtype):
        def dbl(x):
            return x * 2

        got = getattr(self.module.m, name)(dbl, dtype(value))
        assert got == dtype(value) * 2
