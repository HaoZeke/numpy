import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestComplexBoolInout(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine add_real(x)
        complex(kind=8), intent(inout) :: x
        x = x + (1.0d0, 0.0d0)
      end subroutine add_real

      subroutine add_imag(x)
        complex(kind=8), intent(inout) :: x
        x = x + (0.0d0, 1.0d0)
      end subroutine add_imag
    """

    def test_real_part_stays_true(self):
        value = np.array(True)
        self.module.add_real(value)
        assert value.dtype == np.bool_
        assert bool(value.item()) is True

    def test_imag_part_becomes_true(self):
        value = np.array(False)
        self.module.add_imag(value)
        assert value.dtype == np.bool_
        assert bool(value.item()) is True
