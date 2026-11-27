import textwrap

import pytest

from . import util


@pytest.mark.slow
class TestIntentCModuleFunction(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent(
        """\
        module intentcmod
        contains
          integer function plus1(n)
            !f2py intent(c) plus1
            integer, intent(in) :: n
            plus1 = n + 1
          end function plus1
        end module intentcmod
        """
    )

    def test_scalar_result(self):
        assert self.module.intentcmod.plus1(4) == 5
