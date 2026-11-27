import os
import sys
import tempfile

import pytest

from . import util

ROUTINE = """\
      integer function foo()
      foo = 1
      end
"""

MODULE = """\
module foo
contains
  integer function bar()
    bar = 2
  end function bar
end module foo
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32",
                    reason="Fails with MinGW64 Gfortran (Issue #9673)")
@pytest.mark.parametrize("order", ["routine", "module"])
def test_module_and_external_share_a_name(order):
    with tempfile.TemporaryDirectory() as tmp:
        routine = os.path.join(tmp, "foo_routine.f")
        module = os.path.join(tmp, "foo_module.f90")
        with open(routine, "w") as fobj:
            fobj.write(ROUTINE)
        with open(module, "w") as fobj:
            fobj.write(MODULE)
        if order == "routine":
            sources = [routine, module]
        else:
            sources = [module, routine]
        ext = util.build_module(sources, module_name=f"modext_{order}")
        assert ext.foo() == 1
        assert ext.foo.bar() == 2
