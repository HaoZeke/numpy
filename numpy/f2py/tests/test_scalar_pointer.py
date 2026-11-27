import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestScalarPointer(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine incp(f)
  implicit none
  integer, intent(inout) :: f
  pointer :: f
  f = f + 1
end subroutine incp

subroutine incq(g)
  implicit none
  integer, pointer, intent(inout) :: g
  g = g + 3
end subroutine incq
"""

    def test_scalar_pointer_updates(self):
        f = np.array(7, dtype=np.int32)
        self.module.incp(f)
        assert f.item() == 8
        g = np.array(7, dtype=np.int32)
        self.module.incq(g)
        assert g.item() == 10
