import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestTwoSizeExtent(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine shared(a, n, m, seen)
        integer, intent(in) :: n, m
        integer, intent(in) :: a(n, m, n + m)
        integer, intent(out) :: seen
        seen = 42
      end subroutine shared
    """

    def test_rejects_wrong_shared_extent(self):
        assert self.module.shared(np.zeros((2, 3, 5), dtype=np.int32)) == 42
        with pytest.raises(Exception, match=r"m \+ n\) failed"):
            self.module.shared(np.zeros((2, 3, 4), dtype=np.int32))
