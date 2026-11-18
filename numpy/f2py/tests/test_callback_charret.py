from . import util


class TestCallbackCharReturn(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine fill(cb, cb_c, ok)
  integer :: ok
  external cb, cb_c
  character(len=5) :: buf, cbuf
  !f2py intent(out) ok
  !f2py intent(out) buf
  !f2py intent(c, out) cbuf
  call cb(buf)
  call cb_c(cbuf)
  ok = 0
  if (buf == 'hi   ' .and. iachar(cbuf(3:3)) == 0) ok = 1
end subroutine
"""

    def test_callback_string_padding(self):
        assert self.module.fill(lambda: "hi", lambda: "hi") == 1
        assert self.module.fill(lambda: b"hi", lambda: b"hi") == 1
