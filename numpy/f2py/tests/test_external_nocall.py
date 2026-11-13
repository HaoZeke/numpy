import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestExternalNoCall(util.F2PyTest):
    # EXTERNAL that the routine never calls still needs a callback
    # signature. Otherwise the generated C uses an undefined typedef.
    suffix = ".f"
    code = """
      subroutine nocall(f, x, y)
      real f, x, y
      external f
!f2py intent(out) y
      y = x + 1.0
      end
"""

    def test_accepts_callback(self):
        def f(x):
            return x * 10.0

        y = self.module.nocall(f, np.float32(2.0))
        assert y == pytest.approx(3.0)
