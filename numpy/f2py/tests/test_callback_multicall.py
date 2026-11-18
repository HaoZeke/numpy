from . import util


class TestCallbackMultiCall(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine take_two(f, g, x, y)
  integer :: f, g, x, y
  external f, g
  !f2py intent(out) y
  y = f(x) + g(x)
end subroutine
"""

    def test_two_callbacks_in_one_expression(self):
        def f(x):
            return x + 1

        def g(x):
            return x + 2

        assert self.module.take_two(f, g, 3) == 9
