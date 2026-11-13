import numpy as np

from . import util


class TestParameterDimension(util.F2PyTest):
    # The result is declared `y(n)` where `n` is a module parameter.
    # The wrapper must allocate 4 elements, not look up a C variable `n`.
    suffix = ".f90"
    code = """
module parm
  integer, parameter :: n = 4
contains
  function ones(x) result(y)
    real(kind=8), intent(in) :: x
    real(kind=8) :: y(n)
    y = x
  end function ones
end module parm
"""

    def test_result_sized_by_parameter(self):
        y = self.module.parm.ones(np.float64(2.0))
        assert y.shape == (4,)
        assert np.all(y == 2.0)
