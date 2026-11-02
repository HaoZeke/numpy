import platform
import sys

import pytest

import numpy as np
from numpy.f2py import run_main
from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
)

from . import util

IS_PPC_OR_AIX = platform.machine().lower().startswith("ppc") or platform.system() == 'AIX'

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


def test_long_double_array_typedef(tmp_path):
    # a real(16) array needs the long_double typedef, as a scalar does
    fpath = tmp_path / "ldarr.f90"
    fpath.write_text("subroutine twice(a)\n"
                     "  real(16), intent(inout) :: a(2)\n"
                     "  a = 2 * a\n"
                     "end subroutine twice\n")
    # without -h, run_main would leave crackfortran not lowering case
    run_main(["-m", "ldarr", "--lower", "--build-dir", str(tmp_path),
              str(fpath)])
    cmod = (tmp_path / "ldarrmodule.c").read_text()
    assert "typedef long double long_double;" in cmod


@pytest.mark.slow
@pytest.mark.skipif(np.finfo(np.longdouble).precision < 18,
                    reason="long double has fewer than 18 digits")
class TestLongDoubleKind(util.F2PyTest):
    # selected_real_kind(18) is real(10) on x86 and real(16) elsewhere
    sources = [util.getpath("tests", "src", "kind", "long_double.f90")]

    def test_parameters(self):
        # 1 + 2**-60 is a long double that a double rounds to 1
        step = 1 + np.longdouble(2) ** -60
        mod = self.module.long_double_kind
        assert mod.step == step
        assert mod.steps.tolist() == [step, 2 * step]
        assert mod.zstep == step - 1j

    def test_routines(self):
        step = 1 + np.longdouble(2) ** -60
        mod = self.module.long_double_kind
        a = np.array([step, 1])
        mod.twice(a)
        assert a.tolist() == [2 * step, 2]
        assert mod.half(2 * step) == step
        assert mod.zhalf(2 * (step - 1j)) == step - 1j
