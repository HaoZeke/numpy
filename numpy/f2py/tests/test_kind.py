import functools
import platform
import subprocess
import sys
import textwrap

import pytest

import numpy as np
from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
)

from . import util

IS_PPC_OR_AIX = platform.machine().lower().startswith("ppc") or platform.system() == 'AIX'

# REAL(KIND=16) is IEEE binary128 (113-bit significand) or IBM double-double
# (106 bits), so only a long double in one of these formats can hold it.
LONGDOUBLE_HOLDS_REAL16 = np.finfo(np.longdouble).nmant + 1 in (113, 106)


@functools.cache
def has_real16_compiler():
    code = "subroutine real16(x)\n  real(kind=16) :: x\nend subroutine\n"
    return util.check_language("fortran", code, require_snippet=True)

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


@pytest.mark.slow
class TestReal16(util.F2PyTest):
    # gh-9727
    sources = [util.getpath("tests", "src", "kind", "gh9727.f90")]

    def setup_method(self):
        if not has_real16_compiler():
            pytest.skip("Fortran compiler does not support REAL(KIND=16)")
        super().setup_method()

    @pytest.mark.skipif(not LONGDOUBLE_HOLDS_REAL16,
                        reason="long double lacks the REAL(KIND=16) format")
    def test_real16_long_double(self):
        assert self.module.f_get_pi() == np.pi
        assert self.module.f_deg2rad(181.0) == pytest.approx(np.deg2rad(181.0))

    @pytest.mark.skipif(LONGDOUBLE_HOLDS_REAL16,
                        reason="long double has the REAL(KIND=16) format")
    def test_real16_unsupported(self):
        with pytest.raises(NotImplementedError,
                           match=r"f_get_pi: REAL\(KIND=16\)"):
            self.module.f_get_pi()
        with pytest.raises(NotImplementedError,
                           match=r"f_deg2rad: REAL\(KIND=16\)"):
            self.module.f_deg2rad(181.0)


def test_real16_check_only_for_kind16(tmp_path):
    # gh-9727: kinds that are C long double by definition, and C routines in
    # signature files, are called without the REAL(KIND=16) check
    src = tmp_path / "gh9727_kinds.f90"
    src.write_text(textwrap.dedent("""\
        subroutine r16(x)
            real(kind=16), intent(out) :: x
            x = 1
        end subroutine
        subroutine c16(z)
            complex(kind=16), intent(out) :: z
            z = 1
        end subroutine
        subroutine cld(x)
            use iso_c_binding, only: c_long_double
            real(kind=c_long_double), intent(out) :: x
            x = 1
        end subroutine
        """), encoding="ascii")
    pyf = tmp_path / "gh9727_c.pyf"
    pyf.write_text(textwrap.dedent("""\
        python module gh9727_c
        usercode '''
        long double t16(long double value) { return value; }
        '''
        interface
          function t16(value)
            real*16 intent(c) :: t16, value
          end
        end interface
        end python module gh9727_c
        """), encoding="ascii")
    for args in (["-m", "gh9727_kinds", str(src)], [str(pyf)]):
        subprocess.run([sys.executable, "-m", "numpy.f2py", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    text = (tmp_path / "gh9727_kindsmodule.c").read_text(encoding="utf-8")
    assert "gh9727_kinds.r16: REAL(KIND=16)" in text
    assert "gh9727_kinds.c16: REAL(KIND=16)" in text
    assert "gh9727_kinds.cld: REAL(KIND=16)" not in text
    text = (tmp_path / "gh9727_cmodule.c").read_text(encoding="utf-8")
    assert "F2PY_REAL16_IS_LONG_DOUBLE" not in text
