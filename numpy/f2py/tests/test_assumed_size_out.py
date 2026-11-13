import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestAssumedSizeOut(util.F2PyTest):
    # intent(out) assumed-size used to be allocated with dimension -1.
    suffix = ".f90"
    code = """
subroutine fill(n, a)
  integer, intent(in) :: n
  real, intent(out) :: a(*)
  integer :: i
  do i = 1, n
    a(i) = i
  end do
end subroutine
"""

    def test_caller_provides_the_array(self):
        a = np.zeros(3, dtype=np.float32)
        self.module.fill(3, a)
        assert a.tolist() == pytest.approx([1.0, 2.0, 3.0])
