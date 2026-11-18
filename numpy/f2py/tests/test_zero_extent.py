import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestZeroExtent(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine zero(a, size_a)
        integer, intent(in) :: a(0)
        integer, intent(out) :: size_a
        size_a = size(a)
      end subroutine zero

      subroutine square(a, n, size_a)
        integer, intent(in) :: n
        integer, intent(in) :: a(n * n)
        integer, intent(out) :: size_a
        size_a = size(a)
      end subroutine square

      subroutine grid(a, size_a)
        integer, intent(in) :: a(0, 2)
        integer, intent(out) :: size_a
        size_a = size(a)
      end subroutine grid
    """

    def test_zero_extent(self):
        empty = np.empty(0, dtype=np.int32)
        assert self.module.zero(empty) == 0
        assert self.module.square(empty, 0) == 0
        assert self.module.grid(np.empty((0, 2), dtype=np.int32)) == 0
        with pytest.raises(ValueError):
            self.module.zero(np.ones(1, dtype=np.int32))
        with pytest.raises(ValueError):
            self.module.zero(np.array(1, dtype=np.int32))
