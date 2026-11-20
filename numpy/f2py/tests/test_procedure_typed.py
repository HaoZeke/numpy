import subprocess
import sys

import pytest

import numpy as np
from numpy.testing import HAS_SUBPROCESSES

from . import util

_SRC = """\
module m
contains
  real(8) function apply_r(f, x)
    procedure(real(8)) :: f
    real(8), intent(in) :: x
    apply_r = f(x)
  end function apply_r

  integer(8) function apply_i(f, n)
    procedure(integer(8)) :: f
    integer(8), intent(in) :: n
    if (f(n) > 0_8) then
      apply_i = f(n)
    else
      apply_i = 0_8
    end if
  end function apply_i
end module m
"""


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
@pytest.mark.slow
@pytest.mark.parametrize("use_pyf", [False, True])
def test_typed_procedure_dummy(tmp_path, use_pyf):
    src = tmp_path / "proc_typed.f90"
    src.write_text(_SRC)
    if use_pyf:
        pyf = tmp_path / "proc_typed.pyf"
        modname = util.get_temp_module_name()
        subprocess.run(
            [sys.executable, "-m", "numpy.f2py", "-h", str(pyf),
             "-m", modname, str(src), "--overwrite-signature"],
            check=True,
        )
        sig = pyf.read_text()
        assert "external f" in sig
        mod = util.build_module([str(pyf), str(src)], module_name=modname)
    else:
        mod = util.build_module([str(src)])
    assert mod.m.apply_r(lambda x: x * 2.5, 4.0) == pytest.approx(10.0)
    big = np.int64(10**12)
    assert mod.m.apply_i(lambda n: n * 3, big) == 3 * big
