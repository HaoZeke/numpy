import numpy as np
import pytest

from . import util


@pytest.mark.slow
class TestCharInteriorNul(util.F2PyTest):
    # Fortran character values are counted byte strings. A NUL before
    # the end of the value has to survive the trip into and out of the
    # wrapper. Trailing blanks are still dropped on the way out.
    suffix = ".f90"
    code = """
subroutine echo_null(n, buf, code)
  integer, intent(in) :: n
  character(len=n), intent(in) :: buf
  integer, intent(out) :: code
  code = 0
  if (n >= 1 .and. buf(1:1) == 'a') code = code + 1
  if (n >= 2 .and. buf(2:2) == achar(0)) code = code + 2
  if (n >= 3 .and. buf(3:3) == 'b') code = code + 4
end subroutine echo_null

subroutine make_null(buf)
  character(len=3), intent(out) :: buf
  buf(1:1) = 'a'
  buf(2:2) = achar(0)
  buf(3:3) = 'b'
end subroutine make_null

subroutine make_nul_only(buf)
  character(len=3), intent(out) :: buf
  buf = achar(0) // achar(0) // achar(0)
end subroutine make_nul_only

subroutine trail_pad(buf)
  character(len=4), intent(out) :: buf
  buf = 'ab'
end subroutine trail_pad

subroutine inout_null(buf)
  character(len=4), intent(inout) :: buf
  if (buf(1:1) == 'a' .and. buf(2:2) == achar(0) .and. buf(3:3) == 'b') then
    buf = 'ZQ'
  else
    buf = 'NO'
  end if
end subroutine inout_null
"""

    def test_interior_nul_round_trip(self):
        raw = b"a\x00b"
        assert self.module.echo_null(3, raw) == 7
        assert self.module.echo_null(3, "a\x00b") == 7
        assert self.module.echo_null(3, np.array(raw)) == 7
        assert self.module.make_null() == raw
        assert self.module.make_nul_only() == b"\x00\x00\x00"
        assert self.module.trail_pad() == b"ab"
        arr = np.array(b"a\x00b\x00")
        self.module.inout_null(arr)
        assert bytes(arr) == b"ZQ\x00\x00"
