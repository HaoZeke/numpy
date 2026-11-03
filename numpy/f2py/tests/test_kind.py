import platform
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest

import numpy as np
from numpy.f2py import crackfortran, run_main
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


def _compiled(value):
    # a module parameter as f2py exposes the compiler's copy of it
    value = np.asarray(value)[()]
    if isinstance(value, np.bytes_):
        return value.decode()
    if isinstance(value, np.floating):
        return Fraction(*value.as_integer_ratio())
    return int(value)


def _cracked(value):
    # a parameter as crackfortran evaluates it
    if isinstance(value, crackfortran._Real):
        return value.value
    if isinstance(value, (str, crackfortran._Logical)):
        return value if isinstance(value, str) else int(value.value)
    return int(value)


@pytest.mark.slow
class TestParameterValues(util.F2PyTest):
    # parameters are evaluated with Fortran's types and kinds
    sources = [util.getpath("tests", "src", "kind", "parameter_values.f90")]

    def test_compiler_values(self):
        crackfortran.crackfortran([str(self.sources[0])])
        params = crackfortran.f90moduleparams["parameter_values"]
        mod = self.module.parameter_values
        assert {n: _compiled(getattr(mod, n)) for n in params} == {
            n: _cracked(v) for n, v in params.items()}


@pytest.mark.slow
class TestInquiryParameters(util.F2PyTest):
    # inquiries give the model numbers of the kinds of gfortran and flang
    sources = [util.getpath("tests", "src", "kind", "inquiry.f90")]

    def test_compiler_values(self):
        crackfortran.crackfortran([str(self.sources[0])])
        params = crackfortran.f90moduleparams["inquiry"]
        mod = self.module.inquiry
        assert {n: _compiled(getattr(mod, n)) for n in params} == {
            n: _cracked(v) for n, v in params.items()}


def _compiles(code):
    # whether the Fortran compiler meson finds compiles code, which
    # util.check_language does not tell
    with tempfile.TemporaryDirectory() as d:
        Path(d, "meson.build").write_text(
            "project('probe', 'fortran')\n"
            f"assert(meson.get_compiler('fortran').compiles('''{code}'''))\n")
        try:
            run = subprocess.run(["meson", "setup", "build"], cwd=d,
                                 capture_output=True, check=False)
        except OSError:
            return False
    return run.returncode == 0


def _binary128(x):
    # the two 64-bit words, low first, of a positive normal binary128
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while x < Fraction(2) ** e:
        e -= 1
    while x >= Fraction(2) ** (e + 1):
        e += 1
    bits = (e + 16383) << 112 | int((x / Fraction(2) ** e - 1) * 2 ** 112)
    return [w - 2 ** 64 if w >= 2 ** 63 else w
            for w in (bits % 2 ** 64, bits >> 64)]


@pytest.mark.slow
class TestQuadInquiryParameters(util.F2PyTest):
    # numpy has no binary128 on x86-64, so its bits are compared
    sources = [util.getpath("tests", "src", "kind", "inquiry_quad.f90")]

    @classmethod
    def setup_class(cls):
        if not _compiles("real(16) :: x\nend"):
            pytest.skip("the Fortran compiler has no real(16)")
        super().setup_class()

    def test_compiler_values(self):
        crackfortran.crackfortran([str(self.sources[0])])
        params = crackfortran.f90moduleparams["inquiry_quad"]
        mod = self.module.inquiry_quad
        reals = ["huge_q", "tiny_q", "epsilon_q"]
        assert {n: _compiled(getattr(mod, n)) for n in params
                if n not in reals} == {
            n: _cracked(v) for n, v in params.items() if n not in reals}
        bits = [b.tolist() for b in mod.bits()]
        assert bits == [_binary128(params[n].value) for n in reals]


@pytest.mark.slow
class TestKindSelectors(util.F2PyTest):
    # an argument whose kind an intrinsic gives gets the C type of the kind
    sources = [util.getpath("tests", "src", "kind", "kind_selectors.f90")]

    def test_routine(self):
        # one past a C int, so the argument has to be integer(8)
        wide = np.iinfo(np.int32).max + 1
        y, j, w = self.module.kind_selectors.twice(1.1, wide, 1.1 + 2.2j)
        assert (y, j, w) == (2 * 1.1, 2 * wide, 2 * (1.1 + 2.2j))


@pytest.mark.slow
class TestSelectedKindParameters(util.F2PyTest):
    # module parameters are copies of the compiler's values
    sources = [util.getpath("tests", "src", "kind", "selected_kinds.f90")]

    def test_compiler_values(self):
        crackfortran.crackfortran([str(self.sources[0])])
        params = crackfortran.f90moduleparams["selected_kinds"]
        mod = self.module.selected_kinds
        assert {n: int(getattr(mod, n)) for n in params} == params


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
