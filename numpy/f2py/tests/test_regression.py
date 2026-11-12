import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

import numpy as np
import numpy.testing as npt
from numpy.f2py._backends._meson import MesonTemplate
from numpy.testing._private.utils import requires_memory

from . import util

LEN_BEYOND_INT = np.iinfo(np.intc).max + 1


@pytest.mark.slow
class TestIntentInOut(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    def test_inout(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert np.allclose(x, [3, 1, 2])


@pytest.mark.slow
class TestDataOnlyMultiModule(util.F2PyTest):
    # Check that modules without subroutines work
    sources = [util.getpath("tests", "src", "regression", "datonly.f90")]

    def test_mdat(self):
        assert self.module.datonly.max_value == 100
        assert self.module.dat.max_ == 1009
        int_in = 5
        assert self.module.simple_subroutine(5) == 1014


@pytest.mark.slow
class TestModuleWithDerivedType(util.F2PyTest):
    # Check that modules with derived types work
    sources = [util.getpath("tests", "src", "regression", "mod_derived_types.f90")]

    def test_mtypes(self):
        assert self.module.no_type_subroutine(10) == 110
        assert self.module.type_subroutine(10) == 210


@pytest.mark.slow
class TestNegativeBounds(util.F2PyTest):
    # Check that negative bounds work correctly
    sources = [util.getpath("tests", "src", "negative_bounds", "issue_20853.f90")]

    def test_negbound(self):
        xvec = np.arange(12)
        xlow = -6
        xhigh = 4

        # Calculate the upper bound,
        # Keeping the 1 index in mind

        def ubound(xl, xh):
            return xh - xl + 1
        rval = self.module.foo(is_=xlow, ie_=xhigh,
                        arr=xvec[:ubound(xlow, xhigh)])
        expval = np.arange(11, dtype=np.float32)
        assert np.allclose(rval, expval)


@pytest.mark.slow
class TestNumpyVersionAttribute(util.F2PyTest):
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    def test_numpy_version_attribute(self):

        # Check that self.module has an attribute named "__f2py_numpy_version__"
        assert hasattr(self.module, "__f2py_numpy_version__")

        # Check that the attribute __f2py_numpy_version__ is a string
        assert isinstance(self.module.__f2py_numpy_version__, str)

        # Check that __f2py_numpy_version__ has the value numpy.__version__
        assert np.__version__ == self.module.__f2py_numpy_version__


def test_include_path():
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ("fortranobject.c", "fortranobject.h"):
        assert fname in fnames_in_dir


@pytest.mark.slow
class TestIncludeFiles(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "incfile.f90")]
    options = [f"-I{util.getpath('tests', 'src', 'regression')}",
               f"--include-paths {util.getpath('tests', 'src', 'regression')}"]

    def test_gh25344(self):
        exp = 7.0
        res = self.module.add(3.0, 4.0)
        assert exp == res

@pytest.mark.slow
class TestF77Comments(util.F2PyTest):
    # Check that comments are stripped from F77 continuation lines
    sources = [util.getpath("tests", "src", "regression", "f77comments.f")]

    def test_gh26148(self):
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        res = self.module.testsub(x1, x2)
        assert res[0] == 8
        assert res[1] == 15

    def test_gh26466(self):
        # Check that comments after PARAMETER directions are stripped
        expected = np.arange(1, 11, dtype=np.float32) * 2
        res = self.module.testsub2()
        npt.assert_allclose(expected, res)

@pytest.mark.slow
class TestF90Continuation(util.F2PyTest):
    # Check that comments are stripped from F90 continuation lines
    sources = [util.getpath("tests", "src", "regression", "f90continuation.f90")]

    def test_gh26148b(self):
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        res = self.module.testsub(x1, x2)
        assert res[0] == 8
        assert res[1] == 15

@pytest.mark.slow
class TestLowerF2PYDirectives(util.F2PyTest):
    # Check variables are cased correctly
    sources = [util.getpath("tests", "src", "regression", "lower_f2py_fortran.f90")]

    def test_gh28014(self):
        self.module.inquire_next(3)
        assert True

@pytest.mark.slow
def test_gh26623():
    # Including libraries with . should not generate an incorrect meson.build
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f90continuation.f90")],
            ["-lfoo.bar"],
            module_name="Blah",
        )
    except RuntimeError as rerr:
        assert "lparen got assign" not in str(rerr)


def test_meson_library_names_use_valid_identifiers():
    mparser = pytest.importorskip("mesonbuild.mparser")
    libraries = ["foo.bar", "scalapack-openmpi", "foo-bar", "foo_bar", "1foo"]
    meson_build = MesonTemplate(
        modulename="Blah",
        sources=[Path("f90continuation.f90")],
        deps=[],
        libraries=libraries,
        library_dirs=[],
        include_dirs=[],
        object_files=[],
        linker_args=[],
        fortran_args=[],
        build_type="debug",
        python_exe=sys.executable,
    ).generate_meson_build()

    mparser.Parser(meson_build, "meson.build").parse()
    for lib in libraries:
        assert f"declare_dependency(link_args : ['-l{lib}'])" in meson_build

    assert "\nfoo.bar =" not in meson_build
    assert "\nscalapack-openmpi =" not in meson_build
    assert "\nfoo-bar =" not in meson_build
    assert "\n1foo =" not in meson_build


@pytest.mark.slow
@pytest.mark.skipif(platform.system() == "Windows", reason='Unsupported on this platform for now')
def test_gh25784():
    # Compile dubious file using passed flags
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f77fixedform.f95")],
            options=[
                # Meson will collect and dedup these to pass to fortran_args:
                "--f77flags='-ffixed-form -O2'",
                "--f90flags=\"-ffixed-form -g\"",
            ],
            module_name="Blah",
        )
    except ImportError as rerr:
        assert "unknown_subroutine_" in str(rerr)


@pytest.mark.slow
def test_build_rejects_imported_module_name():
    # A second build under an imported name would import as the first one
    code = """
      subroutine which(k)
      integer k
cf2py intent(out) k
      k = {}
      end
"""
    name = "_test_regression_reused_name_ext_module"
    mod = util.build_code(code.format(1), module_name=name)
    assert mod.which() == 1
    with pytest.raises(RuntimeError, match="already in use"):
        util.build_code(code.format(2), module_name=name)


@pytest.mark.slow
class TestComplexStructCompat(util.F2PyTest):
    # Check that .r/.i field access works on complex_double pointers in
    # callstatements (scipy compatibility, gh-30966 follow-up)
    sources = [
        util.getpath("tests", "src", "regression", "complex_struct_compat.pyf"),
        util.getpath("tests", "src", "regression", "complex_struct_compat.f90"),
    ]
    module_name = "_complex_struct_compat_test"

    def test_complex_struct_field_access(self):
        c = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        self.module.zero_imag(c)
        npt.assert_array_equal(c.imag, [0.0, 0.0, 0.0])
        npt.assert_array_equal(c.real, [1.0, 3.0, 5.0])


@pytest.mark.slow
class TestAssignmentOnlyModules(util.F2PyTest):
    # Ensure that variables are exposed without functions or subroutines in a module
    sources = [util.getpath("tests", "src", "regression", "assignOnlyModule.f90")]

    def test_gh27167(self):
        assert (self.module.f_globals.n_max == 16)
        assert (self.module.f_globals.i_max == 18)
        assert (self.module.f_globals.j_max == 72)


@pytest.mark.slow
class TestArrayInitValue(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh18767.pyf")]
    module_name = "gh18767_init"

    def test_myrange(self):
        npt.assert_array_equal(self.module.myrange(5), np.arange(5.0))

    @pytest.mark.skipif(not npt.IS_64BIT,
                        reason="needs arrays larger than a C int")
    @requires_memory(free_bytes=LEN_BEYOND_INT)
    def test_init_beyond_int(self):
        a = self.module.ones(LEN_BEYOND_INT)
        assert a.shape == (LEN_BEYOND_INT,)
        assert np.count_nonzero(a) == LEN_BEYOND_INT


def _wrap_f90(funcwrappers2):
    """Reproduce the F90 line-wrapping logic from rules.py buildmodule."""
    lines = []
    for l in ('\n\n'.join(funcwrappers2) + '\n').split('\n'):
        if 0 <= l.find('!') < 72:
            lines.append(l + '\n')
        elif len(l) > 72 and l[0] == ' ':
            lines.append(l[:72] + '&\n     &')
            l = l[72:]
            while len(l) > 66:
                lines.append(l[:66] + '&\n     &')
                l = l[66:]
            lines.append(l + '\n')
        else:
            lines.append(l + '\n')
    return ''.join(lines).replace('&\n     &\n', '\n')


def _wrap_f77(funcwrappers):
    """Reproduce the F77 line-wrapping logic from rules.py buildmodule."""
    lines = []
    for l in ('\n\n'.join(funcwrappers) + '\n').split('\n'):
        if 0 <= l.find('!') < 66:
            lines.append(l + '\n')
        elif l and l[0] == ' ':
            while len(l) > 66:
                lines.append(l[:66] + '\n     &')
                l = l[66:]
            lines.append(l + '\n')
        else:
            lines.append(l + '\n')
    return ''.join(lines).replace('\n     &\n', '\n')


class TestLineWrapping:
    """Test that F77/F90 line wrapping does not produce invalid continuations.

    Regression tests for gh-28474.
    """

    def test_f90_no_empty_continuation(self):
        # Build a line that is exactly 72 + 66 = 138 chars (spaces included)
        # so the remainder after the first split is exactly 66 chars,
        # which previously produced an empty continuation line with a
        # trailing ampersand on the preceding line.
        name = 'a' * 120
        wrapper = f'      subroutine f2pywrap_{name}()'
        result = _wrap_f90([wrapper])
        for line in result.split('\n'):
            stripped = line.rstrip()
            # No line should end with & unless followed by a continuation
            if stripped.endswith('&'):
                # Find this line's position and check next line exists
                idx = result.index(line + '\n')
                rest = result[idx + len(line) + 1:]
                assert rest.startswith('     &'), (
                    f"Line ends with '&' but next line is not a continuation:\n"
                    f"  {stripped!r}"
                )

    def test_f90_long_line_wraps_correctly(self):
        # A line longer than 72 + 66 chars should wrap without issues
        name = 'b' * 200
        wrapper = f'      subroutine f2pywrap_{name}()'
        result = _wrap_f90([wrapper])
        assert '&\n     &\n' not in result
        # Each line should be at most 73 chars (72 content + trailing &)
        for line in result.split('\n'):
            assert len(line) <= 73, f"Line too long: {line!r}"

    def test_f77_no_empty_continuation(self):
        # A line of exactly 66 chars should not enter the wrapping loop
        line_66 = ' ' + 'x' * 65  # 66 chars, starts with space
        result = _wrap_f77([line_66])
        assert '\n     &' not in result, (
            "Line of exactly 66 chars should not be wrapped"
        )

    def test_f77_boundary_67(self):
        # A line of 67 chars should wrap into 66 + 1
        line_67 = ' ' + 'y' * 66  # 67 chars
        result = _wrap_f77([line_67])
        assert '\n     &' in result
        for line in result.strip().split('\n'):
            assert len(line) <= 66 or line.startswith('     &')


def test_gh22511_parameter_is_copied_to_saved_variable():
    # Flang passes PARAMETER addresses as stack temporaries. The init
    # wrapper must copy them into saved locals (gh-22511).
    from numpy.f2py import f90mod_rules
    from numpy.f2py.crackfortran import crackfortran

    src = util.getpath("tests", "src", "regression", "gh22511.f90")
    blocks = crackfortran([str(src)])
    _, hooks = f90mod_rules.buildhooks(blocks[0])
    assert "integer, save :: f2py_copy_my_const" in hooks
    assert "f2py_copy_my_const = my_const" in hooks
    assert "real(kind=8), save :: f2py_copy_pi" in hooks
    assert "f2py_copy_pi = pi" in hooks
    assert "integer, dimension(2), save :: f2py_copy_arr" in hooks
    assert "f2py_copy_arr = arr" in hooks
    assert "character(len=5), save :: f2py_copy_word" in hooks
    assert "f2py_copy_word = word" in hooks
    assert "f2py_copy_my_var" not in hooks
    call = hooks.split("call f2pysetupfunc(", 1)[1].split(")", 1)[0]
    args = [item.strip() for item in call.split(",")]
    assert "f2py_copy_my_const" in args
    assert "f2py_copy_pi" in args
    assert "f2py_copy_arr" in args
    assert "f2py_copy_word" in args
    assert "my_var" in args
    assert "f2py_copy_my_var" not in args


@pytest.mark.slow
class TestGh22511ParameterCopy(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh22511.f90")]

    def test_parameter_values_and_mutable_variable(self):
        mod = self.module.gh22511
        assert mod.my_const == 1234
        assert mod.pi == 3.0
        assert np.array_equal(np.asarray(mod.arr).reshape(-1), [2, 4])
        assert np.asarray(mod.word).tobytes().split(b"\x00", 1)[0] == b"hello"
        assert mod.my_var == 7
        mod.my_var = 11
        assert mod.my_var == 11
        c = mod.cross(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        assert np.allclose(c, [0.0, 0.0, 1.0])


@pytest.mark.slow
def test_gh9727_kind16_return(tmp_path):
    # Calling the routine in-process segfaults before the fix, so build and
    # call it in a child interpreter.
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    src = util.getpath("tests", "src", "regression", "gh9727.f90")
    (tmp_path / "gh9727.f90").write_text(src.read_text())
    code = """
import math, sys
sys.path.insert(0, ".")
sys.argv = ["f2py", "-c", "-m", "gh9727", "gh9727.f90",
            "--backend", "meson", "--freethreading-compatible"]
from numpy.f2py.f2py2e import main
main()
import gh9727
pi = gh9727.get_pi()
e = gh9727.get_e()
if abs(pi - math.pi) > 1e-12 or abs(e - 1.0) > 1e-12:
    raise SystemExit(f"bad values pi={pi!r} e={e!r}")
"""
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + "\n" + proc.stderr
    # atan(1.0_16) calls _FortranAAtanF128. Some flang packages compile
    # that call but do not ship the symbol.
    if proc.returncode != 0 and "_FortranAAtanF128" in combined:
        pytest.skip("Fortran runtime has no real128 atan")
    assert proc.returncode == 0, combined


@pytest.mark.slow
def test_gh30804_opt_openmp_flag(tmp_path):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    src = util.getpath("tests", "src", "regression", "gh30804.f90")
    (tmp_path / "gh30804.f90").write_text(src.read_text())
    code = """
import sys
sys.path.insert(0, ".")
sys.argv = ["f2py", "-c", "-m", "gh30804", "--opt=-fopenmp",
            "gh30804.f90", "--backend", "meson",
            "--freethreading-compatible"]
from numpy.f2py.f2py2e import main
main()
import gh30804
flag = gh30804.uses_openmp()
if flag != 1:
    raise SystemExit(f"flag={flag!r}")

"""
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


def _assert_fortran_interface(text):
    assert text
    assert "required" not in text
    assert "check(" not in text
    assert "depend(" not in text
    assert "real*8, dimension" in text


def test_gh13553_interface_omits_f2py_attrs():
    from numpy.f2py.crackfortran import crack2fortrangen, crackfortran

    src = util.getpath("tests", "src", "regression", "gh13553.f90")
    block = crackfortran([str(src)])[0]
    _assert_fortran_interface(
        crack2fortrangen(block, "\n      ", as_interface=True))


def test_gh13553_pyf_saved_interface(tmp_path, monkeypatch):
    from numpy.f2py.crackfortran import crackfortran
    from numpy.f2py.f2py2e import main as f2pycli

    src = util.getpath("tests", "src", "regression", "gh13553.f90")
    (tmp_path / "gh13553.f90").write_text(src.read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["f2py", "-m", "gh13553", "-h", "gh13553.pyf", "gh13553.f90",
         "--overwrite-signature"])
    f2pycli()
    blocks = crackfortran([str(tmp_path / "gh13553.pyf")])
    routine = blocks[0]["body"][0]["body"][0]
    assert routine["name"] == "trapz"
    _assert_fortran_interface(routine["saved_interface"])


@pytest.mark.slow
def test_gh13553_pyf_then_compile(tmp_path):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    src = util.getpath("tests", "src", "regression", "gh13553.f90")
    (tmp_path / "gh13553.f90").write_text(src.read_text())
    code = """
import sys
sys.path.insert(0, ".")
sys.argv = ["f2py", "-m", "gh13553", "-h", "gh13553.pyf",
            "gh13553.f90", "--overwrite-signature"]
from numpy.f2py.f2py2e import main
main()
sys.argv = ["f2py", "-c", "-m", "gh13553", "gh13553.pyf", "gh13553.f90",
            "--backend", "meson", "--freethreading-compatible"]
main()
import numpy as np
import gh13553
x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 1.0, 0.0])
got = gh13553.trapz(x, y)
if abs(got - 1.0) > 1e-12:
    raise SystemExit(f"trapz={got!r}")
"""
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
@pytest.mark.slow
class TestGH19767(util.F2PyTest):
    # Prefix kind and length must follow `result` (gh-19767).
    sources = [util.getpath("tests", "src", "regression", "gh19767.f90")]

    def test_real_kind_on_result(self):
        got = self.module.mysqrt(3.5)
        assert abs(got - 12.25) < 1e-12

    def test_character_len_on_result(self):
        assert self.module.greet() == b"hello"

@pytest.mark.slow
def test_gh13356_entry_return(tmp_path):
    # ENTRY inside a function used to raise KeyError while wrapping.
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    src = util.getpath("tests", "src", "regression", "gh13356.f")
    (tmp_path / "gh13356.f").write_text(src.read_text())
    code = """
import sys
sys.path.insert(0, ".")
sys.argv = ["f2py", "-c", "-m", "gh13356", "gh13356.f",
            "--backend", "meson", "--freethreading-compatible"]
from numpy.f2py.f2py2e import main
main()
import gh13356
if abs(gh13356.eta0(2.0) - 2.0) > 1e-12:
    raise SystemExit("eta0")
if abs(gh13356.eta1(2.0, 3.0) - 5.0) > 1e-12:
    raise SystemExit("eta1")
if gh13356.iadd(1.0) != 3:
    raise SystemExit("iadd")
"""
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
