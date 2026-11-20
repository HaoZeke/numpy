import platform
import sys

import pytest

from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
    reset_global_f2py_vars,
)
from numpy.f2py.f2py2e import run_main

from . import util

IS_PPC_OR_AIX = platform.machine().lower().startswith("ppc") or platform.system() == 'AIX'

USED_KIND_CODE = """
module MyKinds
  integer, parameter :: DP = kind(1.d0)
end module MyKinds
module MyOps
  use MyKinds
contains
  function Twice(X) result(Y)
    real(DP), intent(in) :: X
    real(DP) :: Y
    Y = 2 * X
  end function Twice
end module MyOps
"""


@pytest.mark.slow
def test_used_kind_signature_case(tmp_path):
    name = "used_kind_no_lower"
    src = tmp_path / "used_kind.f90"
    src.write_text(USED_KIND_CODE)
    pyf = tmp_path / f"{name}.pyf"
    try:
        run_main(["-h", str(pyf), "-m", name, "--no-lower", str(src)])
    finally:
        reset_global_f2py_vars()
    mod = util.build_module([str(pyf), str(src)], module_name=name)
    assert mod.MyOps.Twice(1.25) == 2.5


@pytest.mark.slow
class TestKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "foo.f90")]

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1,
                        reason="Fails for 32 bit machines")
    def test_int(self):
        """Test `int` kind_func for integers up to 10**40."""
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            assert selectedintkind(i) == selected_int_kind(
                i
            ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"

    def test_real(self):
        """
        Test (processor-dependent) `real` kind_func for real numbers
        of up to 31 digits precision (extended/quadruple).
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"

    @pytest.mark.xfail(IS_PPC_OR_AIX,
                       reason="Some PowerPC may not support full IEEE 754 precision")
    def test_quad_precision(self):
        """
        Test kind_func for quadruple precision [`real(16)`] of 32+ digits .
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32, 40):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"
