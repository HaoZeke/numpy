import subprocess
import sys

import pytest

import numpy as np
from numpy.f2py import crackfortran
from numpy.testing import HAS_SUBPROCESSES

from . import util


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
@pytest.mark.slow
class TestAbstractInterface(util.F2PyTest):
    sources = [util.getpath("tests", "src", "abstract_interface", "foo.f90")]

    skip = ["add1", "add2"]

    def test_abstract_interface(self):
        assert self.module.ops_module.foo(3, 5) == (8, 13)

    def test_parse_abstract_interface(self):
        # Test gh18403
        fpath = util.getpath("tests", "src", "abstract_interface",
                             "gh18403_mod.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert len(mod[0]["body"]) == 1
        assert mod[0]["body"][0]["block"] == "abstract interface"


_ABSTRACT_IFACE_SRC = """\
module m
  abstract interface
    subroutine cb(x)
      real(8), intent(in) :: x
    end subroutine cb
  end interface
contains
  integer function add1(n)
    integer, intent(in) :: n
    add1 = n + 1
  end function add1
end module m

subroutine work(a, s)
  real(8), intent(in) :: a(:)
  real(8), intent(out) :: s
  abstract interface
    real(8) function fn_t(x)
      real(8), intent(in) :: x
    end function fn_t
  end interface
  s = sum(a)
end subroutine work
"""


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
@pytest.mark.slow
@pytest.mark.parametrize("use_pyf", [False, True])
def test_abstract_interface_emit(tmp_path, use_pyf):
    src = tmp_path / "absiface.f90"
    src.write_text(_ABSTRACT_IFACE_SRC)
    sources = [str(src)]
    if use_pyf:
        pyf = tmp_path / "absiface.pyf"
        modname = util.get_temp_module_name()
        subprocess.run(
            [sys.executable, "-m", "numpy.f2py", "-h", str(pyf),
             "-m", modname, str(src), "--overwrite-signature"],
            check=True,
        )
        sig = pyf.read_text()
        code = "\n".join(line.split("!", 1)[0] for line in sig.splitlines())
        assert "unknown_abstract_interface" not in code
        assert "end abstract interface" not in code
        assert "abstract interface" in code
        sources = [str(pyf), str(src)]
        mod = util.build_module(sources, module_name=modname)
    else:
        mod = util.build_module(sources)
    assert mod.m.add1(4) == 5
    assert mod.work(np.array([1.0, 2.0])) == 3.0
