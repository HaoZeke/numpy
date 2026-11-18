from . import util


class TestCallbackCharLiteral(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine take_lit(cb, n)
  integer :: cb, n
  external cb
  !f2py intent(out) n
  n = cb("a,b)", 3)
end subroutine
"""

    def test_quoted_callback_argument(self):
        seen = []

        def cb(text, value):
            seen.append((text, value))
            return 7

        assert self.module.take_lit(cb) == 7
        assert seen == [("a,b)", 3)]
