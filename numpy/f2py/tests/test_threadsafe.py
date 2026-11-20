import pytest

from . import util


@pytest.mark.slow
class TestThreadsafeCallback(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine run_sub(f, x, y)
  external f
  real(8) f, x, y
!f2py threadsafe
!f2py intent(out) y
  y = f(x)
end subroutine

function run_fun(f, x)
  external f
  real(8) f, x, run_fun
!f2py threadsafe
  run_fun = f(x)
end function
"""

    @pytest.mark.parametrize("name", ["run_sub", "run_fun"])
    def test_callback(self, name):
        assert getattr(self.module, name)(lambda x: 2 * x, 3.0) == 6.0
