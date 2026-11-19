import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCharStrArray(util.F2PyTest):
    suffix = ".f90"
    code = """
        subroutine see(c, n, ok)
          character(len=*), intent(in) :: c(2)
          integer, intent(out) :: n, ok
          n = len(c)
          ok = 0
          if (c(1) == 'abc' .and. c(2) == 'def') ok = 1
        end subroutine see

        subroutine edit(c, n)
          character(len=*), intent(inout) :: c(2)
          integer, intent(out) :: n
          n = len(c)
          c(1)(1:1) = 'Z'
        end subroutine edit
        """

    @pytest.mark.parametrize("dtype", ["U", "S"])
    def test_len_and_edit(self, dtype):
        if dtype == "U":
            with pytest.raises(UnicodeEncodeError):
                self.module.see(np.array(["ébc", "def"]))
        src = np.array(["abc", "def"], dtype=dtype)
        n, ok = self.module.see(src)
        assert n == 3 and ok == 1
        buf = np.array(["abc", "def"], dtype=dtype)
        n = self.module.edit(buf)
        assert n == 3
        assert buf[0] == np.array("Zbc", dtype=dtype)[()]
