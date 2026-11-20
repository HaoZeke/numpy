import subprocess
import sys

import pytest

import numpy as np

from . import util

SOURCE = """\
subroutine optdep(n, a, flag)
  integer, intent(in) :: n
  integer, intent(in), optional :: a(n)
  integer, intent(out) :: flag
  if (present(a)) then
    flag = a(1)
  else
    flag = -1
  end if
end subroutine
"""


def _build(tmp_path, use_pyf):
    src = tmp_path / "optdep.f90"
    src.write_text(SOURCE)
    modname = util.get_temp_module_name()
    if use_pyf:
        pyf = tmp_path / "optdep.pyf"
        subprocess.check_call(
            [
                sys.executable, "-m", "numpy.f2py",
                "-h", str(pyf), "-m", modname, str(src),
                "--overwrite-signature",
            ],
            cwd=tmp_path,
        )
        return util.build_module([str(pyf), str(src)], module_name=modname)
    return util.build_module([str(src)], module_name=modname)


@pytest.mark.slow
@pytest.mark.parametrize("use_pyf", [False, True])
def test_omitted_optional_array_uses_bound(tmp_path, use_pyf):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler found")

    mod = _build(tmp_path, use_pyf)
    arr = np.array([7, 8, 9], dtype=np.int32)
    assert mod.optdep(n=3) == 0
    assert mod.optdep(a=arr) == 7
    assert mod.optdep(n=3, a=arr) == 7
    with pytest.raises(Exception, match=r"shape\(a, 0\) == n"):
        mod.optdep(n=4, a=arr)


@pytest.mark.slow
def test_array_bound_without_scalar_keeps_working(tmp_path):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler found")

    src = tmp_path / "cyc.f90"
    src.write_text(
        "subroutine cyc(a, b, flag)\n"
        "  integer, intent(in), optional :: a(3)\n"
        "  integer, intent(in), optional :: b(3)\n"
        "  integer, intent(out) :: flag\n"
        "  flag = 0\n"
        "  if (present(a)) flag = flag + 1\n"
        "  if (present(b)) flag = flag + 10\n"
        "end subroutine\n"
    )
    modname = util.get_temp_module_name()
    pyf = tmp_path / "cyc.pyf"
    pyf.write_text(
        f"python module {modname}\n"
        "interface\n"
        "  subroutine cyc(a, b, flag)\n"
        "    integer, intent(in), optional, dimension(shape(b,0)), depend(b) :: a\n"
        "    integer, intent(in), optional, dimension(shape(a,0)), depend(a) :: b\n"
        "    integer, intent(out) :: flag\n"
        "  end subroutine\n"
        "end interface\n"
        f"end python module {modname}\n"
    )
    mod = util.build_module([str(pyf), str(src)], module_name=modname)
    arr = np.zeros(3, dtype=np.int32)
    assert mod.cyc(b=arr) == 11
    assert mod.cyc(a=arr, b=arr) == 11
