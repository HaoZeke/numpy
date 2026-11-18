import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCharArrayBlanks(util.F2PyTest):
    # Fortran blank-pads a short character array element and treats
    # trailing blanks as insignificant. A NumPy "|S" element is
    # NUL-padded, so the wrapper has to pass blanks and leave the
    # caller's buffer alone.
    suffix = ".f90"
    code = """
subroutine arr_eq(x, flag)
  character(len=*), intent(in) :: x(:)
  integer, intent(out) :: flag
  flag = 0
  if (len(x) == 3) flag = flag + 1
  if (size(x) == 2) flag = flag + 2
  if (x(1) == 'a') flag = flag + 4
  if (x(1) == 'a  ') flag = flag + 8
  if (x(2) == 'bbb') flag = flag + 16
end subroutine arr_eq

"""

    def test_short_elements_match_fortran_blanks(self):
        values = np.array([b"a", b"bbb"])
        raw = values.tobytes()
        assert self.module.arr_eq(values) == 31
        assert values.tobytes() == raw
        assert self.module.arr_eq(["a", "bbb"]) == 31

        spaced = np.array([b"a ", b"bbb"])
        spaced_raw = spaced.tobytes()
        assert self.module.arr_eq(spaced) == 31
        assert spaced.tobytes() == spaced_raw
