import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestHalfInout(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine bump_i(x)
        integer, intent(inout) :: x
        x = x + 1
      end subroutine bump_i

      subroutine bump_r(x)
        real(kind=8), intent(inout) :: x
        x = x + 1.0d0
      end subroutine bump_r

      subroutine bump_c(x)
        complex(kind=8), intent(inout) :: x
        x = x + (1.0d0, 0.0d0)
      end subroutine bump_c
    """

    def test_float16_integer(self):
        value = np.array(np.float16(3.0))
        self.module.bump_i(value)
        assert value.dtype == np.float16
        assert value == np.float16(4.0)

    def test_float16_real(self):
        value = np.array(np.float16(3.5))
        self.module.bump_r(value)
        assert value.dtype == np.float16
        assert value == np.float16(4.5)

    def test_float16_complex(self):
        value = np.array(np.float16(3.5))
        self.module.bump_c(value)
        assert value.dtype == np.float16
        assert value == np.float16(4.5)
