import pytest

from . import util


@pytest.mark.slow
class TestFixedContName(util.F2PyTest):
    suffix = ".f"
    code = """
      integer function fo
     &obar(x)
      integer x
      foobar = x + 7
      end

      integer function real
     &x(a, do
     &it)
      integer a, doit
      realx = a + doit
      end

      real function
     &keepblank(x, msg)
      real x
      character*18 msg
cf2py character*18 optional, intent(in,out) :: msg = 'err; see code
cf2py&here'
      keepblank = x + 1.0
      end
"""

    def test_fixed_continuation_joins_name(self):
        assert self.module.foobar(3) == 10
        assert self.module.realx(a=3, doit=4) == 7
        assert self.module.keepblank(2.0) == (3.0, b"err; see code here")
