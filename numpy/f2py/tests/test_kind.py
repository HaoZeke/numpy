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

# A non-zero status means the import ran a compiler, which this test forbids.
_IMPORT_PROBE_STATUS = 2
# The failure text only needs the tail of the child traceback.
_IMPORT_STDERR_TAIL = 500

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

    def test_real_range(self):
        # either side of each format's precision and range, including
        # flang's real(2) and real(3)
        selectedrealkind = self.module.selectedrealkind
        cases = [(0, 0), (0, 4), (0, 5), (2, 0), (3, 0), (6, 70),
                 (15, 400), (19, 0), (33, 0), (34, 0)]
        for precision, exponent_range in cases:
            got = selectedrealkind(precision, exponent_range)
            expected = selected_real_kind(precision, exponent_range)
            assert got == expected, (
                f"selected_real_kind({precision}, {exponent_range}): "
                f"expected {expected!r} but got {got!r}")

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
    if isinstance(value, np.complexfloating):
        return _compiled(value.real), _compiled(value.imag)
    return int(value)


def _cracked(value):
    # a parameter as crackfortran evaluates it
    if isinstance(value, crackfortran._Real):
        return value.value
    if isinstance(value, crackfortran._Complex):
        return value.real, value.imag
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


# binary128 stores its exponent with a bias of 16383.
_BINARY128_EXP_BIAS = 16383
# A binary128 value is a pair of 64-bit words.
_WORD_BITS = 64


def _binary128(x):
    # the two 64-bit words, low first, of a positive normal binary128
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while x < Fraction(2) ** e:
        e -= 1
    while x >= Fraction(2) ** (e + 1):
        e += 1
    bits = (
        (e + _BINARY128_EXP_BIAS) << crackfortran._BINARY128_NMANT
        | int((x / Fraction(2) ** e - 1) * 2 ** crackfortran._BINARY128_NMANT))
    return [w - 2 ** _WORD_BITS if w >= 2 ** (_WORD_BITS - 1) else w
            for w in (bits % 2 ** _WORD_BITS, bits >> _WORD_BITS)]


@pytest.mark.slow
class TestQuadInquiryParameters(util.F2PyTest):
    # numpy has no binary128 on x86-64, so its bits are compared
    sources = [util.getpath("tests", "src", "kind", "inquiry_quad.f90")]

    @classmethod
    def setup_class(cls):
        if 16 not in crackfortran._real_formats:
            pytest.skip("this compiler's selected_real_kind has no real(16)")
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
class TestInitialValues(util.F2PyTest):
    # C read 1.0d-6 as 1.0 - 6
    sources = [util.getpath("tests", "src", "kind", "initial_values.f90")]

    def test_defaults(self):
        assert self.module.scale(2.0) == 2.0 * 1e-6 * 3


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
    # c_long_double is C long double on every platform. real(16) is not.
    fpath = tmp_path / "ldarr.f90"
    fpath.write_text("subroutine twice(a)\n"
                     "  use, intrinsic :: iso_c_binding, only: c_long_double\n"
                     "  real(c_long_double), intent(inout) :: a(2)\n"
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


def _quad_long_double():
    info = np.finfo(np.longdouble)
    return (info.nmant == crackfortran._BINARY128_NMANT
            or (info.maxexp == crackfortran._IEEE_DOUBLE_MAXEXP
                and info.nmant > crackfortran._BINARY64_NMANT))


def test_unmapped_kind_number_raises():
    from numpy.f2py import capi_maps
    var = {'typespec': 'integer', 'kindselector': {'kind': '16'}}
    with pytest.raises(ValueError, match=r"integer\(kind=16\) 'n'"):
        capi_maps.getctype(var, 'n')
    logical = {'typespec': 'logical', 'kindselector': {'kind': '16'}}
    with pytest.raises(ValueError, match=r"logical\(kind=16\)"):
        capi_maps.getctype(logical, 'flag')
    # a complex kind must not fall through to the byte-count map
    wide = {'typespec': 'complex', 'kindselector': {'kind': '32'}}
    with pytest.raises(ValueError, match=r"complex\(kind=32\)"):
        capi_maps.getctype(wide, 'z')


def test_real16_follows_long_double():
    from numpy.f2py import capi_maps
    var = {'typespec': 'real', 'kindselector': {'kind': '16'}}
    if _quad_long_double():
        assert capi_maps.getctype(var, 'x') == 'long_double'
    else:
        with pytest.raises(ValueError, match=r"real\(kind=16\) 'x'"):
            capi_maps.getctype(var, 'x')


def test_real_star_12_follows_storage():
    from numpy.f2py import capi_maps
    var = {'typespec': 'real', 'kindselector': {'*': '12'}}
    if np.dtype(np.longdouble).itemsize == 12:
        assert capi_maps.getctype(var, 'x') == 'long_double'
    else:
        with pytest.raises(ValueError, match=r"real\(kind=12\) 'x'"):
            capi_maps.getctype(var, 'x')


def test_unknown_kind_reported_once(capsys):
    from numpy.f2py import capi_maps
    var = {'typespec': 'real', 'kindselector': {'kind': 'wp_unknown'}}
    capi_maps._defaulted_kinds.discard(('real', 'wp_unknown'))
    assert capi_maps.getctype(var, 'a') == 'float'
    assert capi_maps.getctype(var, 'b') == 'float'
    # the kind is named in the message and again in the override example
    assert capsys.readouterr().err.count('is mapped to C') == 1


def test_character_kind_other_than_one():
    from numpy.f2py import capi_maps
    var = {'typespec': 'character', 'charselector': {'kind': '4', 'len': '3'}}
    with pytest.raises(ValueError, match=r"character\(kind=4\) 'w'"):
        capi_maps.getctype(var, 'w')
    c_char = {'typespec': 'character', 'charselector': {'kind': 'c_char'}}
    assert capi_maps.getctype(c_char, 'c') == 'string'


def test_kind_names_follow_their_numbers():
    from numpy.f2py import capi_maps
    half = {'typespec': 'real', 'kindselector': {'kind': 'c_float16'}}
    assert capi_maps.getctype(half, 'h') == 'float16'
    named = {'typespec': 'real', 'kindselector': {'kind': 'real16'}}
    assert capi_maps.getctype(named, 'h') == 'float16'
    quad = {'typespec': 'real', 'kindselector': {'kind': 'real128'}}
    if _quad_long_double():
        assert capi_maps.getctype(quad, 'q') == 'long_double'
    else:
        with pytest.raises(ValueError, match=r"real\(kind=real128\) 'q'"):
            capi_maps.getctype(quad, 'q')
    wide = {'typespec': 'integer', 'kindselector': {'kind': 'c_int128_t'}}
    with pytest.raises(ValueError, match=r"integer\(kind=c_int128_t\)"):
        capi_maps.getctype(wide, 'i')


def test_c_size_t_width():
    from numpy.f2py import capi_maps
    var = {'typespec': 'integer', 'kindselector': {'kind': 'c_size_t'}}
    ctype = capi_maps.getctype(var, 'n')
    widths = {'unsigned_char': 1, 'unsigned_short': 2, 'unsigned': 4,
              'unsigned_long_long': 8}
    assert widths[ctype] == np.dtype(np.uintp).itemsize


def test_fast_integer_kinds_are_signed():
    from numpy.f2py import capi_maps
    signed = {'signed_char', 'short', 'int', 'long_long'}
    for name in ('c_int_fast16_t', 'c_int_fast32_t'):
        var = {'typespec': 'integer', 'kindselector': {'kind': name}}
        assert capi_maps.getctype(var, 'n') in signed


def test_half_kind_in_generated_c(tmp_path):
    fpath = tmp_path / "halfarr.f90"
    fpath.write_text("subroutine twice(a)\n"
                     "  real(2), intent(inout) :: a(2)\n"
                     "  a = 2 * a\n"
                     "end subroutine twice\n")
    run_main(["-m", "halfarr", "--lower", "--build-dir", str(tmp_path),
              str(fpath)])
    cmod = (tmp_path / "halfarrmodule.c").read_text()
    assert "typedef _Float16 float16;" in cmod
    assert "NPY_HALF" in cmod


def test_integer16_generation_stops(tmp_path):
    fpath = tmp_path / "big.f90"
    fpath.write_text("module big\n"
                     "  implicit none\n"
                     "  integer(16) :: n = 7\n"
                     "end module big\n")
    with pytest.raises(ValueError, match=r"integer\(kind=16\) 'n'"):
        run_main(["-m", "big", "--lower", "--build-dir", str(tmp_path),
                  str(fpath)])


class TestHostParameters:
    # a module procedure sees its module's parameters
    def test_kind_and_dimension(self, tmp_path):
        src = Path(util.getpath("tests", "src", "kind", "host.f90")).read_text()
        fpath = tmp_path / "host.f90"
        fpath.write_text(src)
        blocks = crackfortran.crackfortran([str(fpath)])
        module = next(b for b in blocks if b.get('name') == 'host')
        routines = {b['name']: b for b in module['body']}
        scale = routines['scale_dp']['vars']['a']
        assert scale['kindselector']['kind'] == '8'
        assert scale['dimension'] == ['3']
        shadowed = routines['scale_arg']['vars']['a']
        assert shadowed['kindselector']['kind'] == '8'
        assert shadowed['dimension'] != ['3']


@pytest.mark.slow
class TestHostRoutine(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "host.f90")]

    def test_scale(self):
        mod = self.module.host
        a = np.array([1.0, 2.0, 3.0])
        mod.scale_dp(a)
        assert a.tolist() == [2.0, 4.0, 6.0]
        b = np.array([1.0, 2.0])
        mod.scale_arg(b, 2)
        assert b.tolist() == [2.0, 4.0]


# 2**40 + 3 does not fit in 32 bits, so a c_size_t round trip must keep it.
_C_SIZE_T_PAST_32_BITS = 2 ** 40 + 3
# 2**40 + 1 does not fit in 32 bits, so an 8-byte fast kind must keep it.
_FAST_KIND_PAST_32_BITS = 2 ** 40 + 1
# 1000 fits in every signed kind f2py maps, for a narrower fast kind.
_FAST_KIND_FITS_ALL = 1000


@pytest.mark.slow
class TestSizeKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "sizet.f90")]

    def test_scalar(self):
        value = _C_SIZE_T_PAST_32_BITS
        assert int(self.module.sizemod.addone(value)) == value + 1


@pytest.mark.slow
class TestFastKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "fastint.f90")]

    def _round_trip(self, func):
        sample = int(func(1))
        width = np.array(sample).dtype.itemsize
        value = (_FAST_KIND_PAST_32_BITS if width >= 8
                 else _FAST_KIND_FITS_ALL)
        assert int(func(value)) == value + 1

    def test_fast16(self):
        self._round_trip(self.module.fastmod.add16)

    def test_fast32(self):
        self._round_trip(self.module.fastmod.add32)


@pytest.mark.slow
class TestHalfKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "half.f90")]

    @classmethod
    def setup_class(cls):
        if 2 not in crackfortran._real_formats:
            pytest.skip("the Fortran compiler has no real(2)")
        super().setup_class()

    def test_value(self):
        mod = self.module.halfmod
        assert mod.x.dtype == np.float16
        assert mod.x == np.float16(1.5)
        a = np.array([1, 2], dtype=np.float16)
        mod.twice(a)
        assert a.tolist() == [np.float16(2), np.float16(4)]
        assert mod.half_of(np.float16(3)) == np.float16(1.5)

    def test_check_message(self):
        with pytest.raises(Exception, match=r"-1") as exc:
            self.module.halfmod.check_positive(np.float16(-1))
        assert "varshowvalue" not in str(exc.value)


def test_import_spawns_no_compiler():
    """Importing f2py runs no compiler, with ``FC`` unset or set to gfortran."""
    import os
    code = (
        "import subprocess\n"
        "def run(*args, **kwargs):\n"
        f"    raise SystemExit({_IMPORT_PROBE_STATUS})\n"
        "subprocess.run = run\n"
        "import numpy.f2py\n"
        "raise SystemExit(0)\n")
    for fc in (None, "gfortran"):
        env = os.environ.copy()
        env.pop("FC", None)
        if fc:
            env["FC"] = fc
        ran = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True)
        assert ran.returncode == 0, (
            fc, ran.returncode, ran.stderr[-_IMPORT_STDERR_TAIL:])
