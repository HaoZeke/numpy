import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCharStarOut(util.F2PyTest):
    suffix = ".f90"
    code = """
        subroutine put_scalar(c)
          character(len=*), intent(out) :: c
          c = 'hello'
        end subroutine put_scalar

        subroutine put_array(c)
          character(len=*), intent(out) :: c(2)
          c(1) = 'ab'
          c(2) = 'cd'
        end subroutine put_array
        """

    @pytest.mark.parametrize("kind", ["scalar", "array"])
    def test_caller_buffer(self, kind):
        if kind == "scalar":
            assert self.module.put_scalar("hello") == b"hello"
        else:
            out = self.module.put_array(np.array([b"xx", b"yy"]))
            assert out[0] == b"ab" and out[1] == b"cd"
