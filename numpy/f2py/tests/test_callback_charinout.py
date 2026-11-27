import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCallbackCharInOut(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine poke(cb, s)
  implicit none
  external cb
  character*(*) :: s
  !f2py intent(inout) s
  call cb(s)
end subroutine poke
"""

    @pytest.mark.parametrize("text", [b"hello", b"abcd"])
    def test_inout_character(self, text):
        buf = np.array(text)
        seen = []

        def cb(view):
            seen.append(view.tobytes())
            view[0] = b"Z"

        assert self.module.poke(cb, buf) is None
        assert seen == [text]
        assert buf.tobytes() == b"Z" + text[1:]
