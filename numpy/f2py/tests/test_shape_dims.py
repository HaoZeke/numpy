import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAssumedShapeResult(util.F2PyTest):
    # The result is an assumed-shape array. Its extent must come from
    # the input array, whose Dims symbol exists. shape() on the function
    # name looks for a Dims symbol that was never declared.
    suffix = ".f90"
    code = """
module m
contains
  function dct_rk(x) result(result)
    real, intent(in) :: x(:)
    real, allocatable :: result(:)
    result = x + 1
  end function
end module
"""

    def test_result_follows_input(self):
        got = np.asarray(self.module.m.dct_rk(np.array([1.0, 2.0, 3.0], dtype=np.float32)))
        assert got.dtype == np.float32
        assert np.allclose(got, [2.0, 3.0, 4.0])
