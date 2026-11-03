import os
import platform
import re
import sys
from pathlib import Path

import pytest

import numpy as np
import numpy.testing as npt
from numpy.f2py._backends._meson import MesonTemplate

from . import util


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
class TestScalarCheckMessage(util.F2PyTest):
    # A failing check() on a scalar raises the module error with the value,
    # not SystemError from PyErr_Format rejecting a printf conversion
    sources = [util.getpath("tests", "src", "regression", "scalar_check.f90")]

    @pytest.mark.parametrize("nth, name, value, check", [
        ("1st", "r8", -1.5, "r8 > 0"),
        ("2nd", "r4", -2.5, "r4 > 0"),
        ("3rd", "i1", -3, "i1 > 0"),
        ("4th", "i2", -4, "i2 > 0"),
        ("5th", "i4", -5, "i4 > 0"),
        ("6th", "i8", -2**40, "i8 > 0"),
        ("7th", "c", b"z", "c == 'a'"),
    ])
    def test_failed_check(self, nth, name, value, check):
        args = {"r8": 1, "r4": 1, "i1": 1, "i2": 1, "i4": 1, "i8": 1, "c": b"a"}
        args[name] = value
        error = getattr(self.module, f"_{self.module.__name__}_error")
        msg = f"({check}) failed for {nth} argument {name}: chk:{name}={value!r}"
        with pytest.raises(error, match=re.escape(msg)):
            self.module.chk(**args)


@pytest.mark.slow
class TestNpyIntegerTypes(util.F2PyTest):
    # The unsigned kinds and long_long map to the npy_common.h types (gh-21161)
    sources = [
        util.getpath("tests", "src", "regression", "gh21161.pyf"),
        util.getpath("tests", "src", "regression", "gh21161.f90"),
    ]
    module_name = "_gh21161_test"

    def test_unsigned_arrays_and_long_long(self):
        a = np.array([1, 2**8 - 2], dtype=np.uint8)
        b = np.array([1, 2**16 - 2], dtype=np.uint16)
        c = np.array([1, 2**32 - 2], dtype=np.uint32)
        d = np.array([1, 2**63], dtype=np.uint64)
        assert self.module.incr(a, b, c, d, 2**40) == 2**40 + 1
        npt.assert_array_equal(a, [2, 2**8 - 1])
        npt.assert_array_equal(b, [2, 2**16 - 1])
        npt.assert_array_equal(c, [2, 2**32 - 1])
        npt.assert_array_equal(d, [2, 2**63 + 1])


def test_gh21161_typedefs_from_npy_common(tmp_path):
    # f2py's integer type names alias the npy_common.h types, without
    # platform branches of their own
    pyf = util.getpath("tests", "src", "regression", "gh21161.pyf")
    ret = np.f2py.run_main([str(pyf), "--build-dir", str(tmp_path)])
    csrc = Path(ret["_gh21161_test"]["csrc"][0]).read_text()
    assert "#include <numpy/npy_common.h>" in csrc
    for typedef in ["typedef npy_ubyte unsigned_char;",
                    "typedef npy_ushort unsigned_short;",
                    "typedef npy_longlong long_long;",
                    "typedef npy_ulonglong unsigned_long_long;"]:
        assert csrc.count(typedef) == 1
    assert "__int64" not in csrc


@pytest.mark.slow
def test_debugcapi_long_long_format(capfd, monkeypatch):
    # --debug-capi prints long_long with NPY_LONGLONG_FMT, not %ld
    monkeypatch.setenv("CFLAGS", f"{os.environ.get('CFLAGS', '')} -Werror=format")
    module = util.build_module(
        [util.getpath("tests", "src", "regression", "scalar_check.f90")],
        options=["--debug-capi"],
        module_name="_debugcapi_long_long",
    )
    capfd.readouterr()
    module.chk(1, 1, 1, 1, 1, 2**40, b"a")
    assert f"debug-capi:i8={2**40}\n" in capfd.readouterr().err


@pytest.mark.slow
class TestAssignmentOnlyModules(util.F2PyTest):
    # Ensure that variables are exposed without functions or subroutines in a module
    sources = [util.getpath("tests", "src", "regression", "assignOnlyModule.f90")]

    def test_gh27167(self):
        assert (self.module.f_globals.n_max == 16)
        assert (self.module.f_globals.i_max == 18)
        assert (self.module.f_globals.j_max == 72)


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
