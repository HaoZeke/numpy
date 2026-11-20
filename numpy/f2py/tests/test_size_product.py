import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestSizeProduct(util.F2PyTest):
    suffix = ".f90"
    code = """
        subroutine sqsize(a, n)
          integer, intent(in) :: n
          real, intent(in) :: a(n * n)
        end subroutine sqsize
        """

    @pytest.mark.parametrize("n, length", [(92682, 18532), (100000, 1)])
    def test_wrapped_length_rejected(self, n, length):
        a = np.ones(length, dtype=np.float32)
        with pytest.raises(ValueError, match=str(n * n)):
            self.module.sqsize(a, n)
