import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestSizeSection(util.F2PyTest):
    # size(a(1, :)) is the second extent. size(a(:))/2 is half the vector.
    suffix = ".f90"
    code = """
module m
contains
  function row(a) result(b)
    real, intent(in) :: a(:,:)
    real :: b(size(a(1, :)))
    b = a(1, :)
  end function
  function half(a) result(b)
    real, intent(in) :: a(:)
    real :: b(size(a(:)) / 2)
    b = a(1:size(b))
  end function
end module
"""

    @pytest.mark.parametrize("which,data,expect", [
        ("row", [[1, 2, 3], [4, 5, 6]], [1, 2, 3]),
        ("half", [10, 20, 30, 40, 50], [10, 20]),
    ])
    def test_extent(self, which, data, expect):
        arr = np.array(data, dtype=np.float32, order="F")
        out = getattr(self.module.m, which)(arr)
        assert out.shape == (len(expect),)
        assert np.allclose(out, np.array(expect, dtype=np.float32))
