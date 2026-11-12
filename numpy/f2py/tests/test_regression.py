import copy
import json
import errno
import os
import pickle
import platform
import subprocess
import shutil
import sys
import warnings
import tempfile
from pathlib import Path

import pytest

import numpy as np
import numpy.testing as npt
from numpy.f2py._backends._meson import MesonBackend, MesonTemplate
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
class TestIntentInOutScalar(util.F2PyTest):
    # gh-20053: write-back of scalar intent(inout) arguments
    sources = [util.getpath("tests", "src", "regression", "inout_scalar.f90")]

    @pytest.mark.parametrize(
        "dtype", ["f8", ">f8", "f4", "f2", "i8", ">i4", "u1", "?", "O"])
    def test_write_back(self, dtype):
        x = np.array(1, dtype=dtype)
        self.module.inc_d(x)
        assert x.dtype == dtype
        assert x == np.array(2, dtype=dtype)

    def test_not_an_array(self):
        # Nothing to write back to, but the call works
        lst = [1.0]
        self.module.inc_d(1.0)
        self.module.inc_d(lst)
        assert lst == [1.0]

    def test_first_element(self):
        x = np.array([1.0, 5.0])
        self.module.inc_d(x)
        npt.assert_array_equal(x, [2.0, 5.0])

    def test_unsafe_cast(self):
        x = np.array(255, dtype=np.uint8)
        self.module.inc_i8(x)
        assert x == 0

    def test_read_only(self):
        x = np.array(1.0)
        x.flags.writeable = False
        with pytest.raises(ValueError, match="read-only"):
            self.module.inc_d(x)
        assert x == 1.0

    @pytest.mark.skipif(not npt.HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_object_array_refcount(self):
        x = np.array(1.0, dtype=object)
        self.module.inc_d(x)
        value = x[()]
        del x
        fresh = float("2.0")
        refs, fresh_refs = sys.getrefcount(value), sys.getrefcount(fresh)
        assert value == fresh
        assert refs == fresh_refs

    @pytest.mark.parametrize("dtype", ["c16", ">c16", "c8"])
    def test_complex(self, dtype):
        x = np.array(1 + 1j, dtype=dtype)
        self.module.scale_z(x)
        assert x == 2 + 2j

    def test_complex_to_real(self):
        # Only the real part is written back, without a ComplexWarning
        x = np.array(2.0)
        self.module.scale_z(x)
        assert x == 4.0

    def test_complex_to_bool(self):
        x = np.array(True)
        self.module.scale_z(x)
        assert x


@pytest.mark.slow
class TestIntentInOutInteger1(util.F2PyTest):
    # gh-20053: the integer(1) write-back did not compile
    sources = [
        util.getpath("tests", "src", "regression", "inout_scalar_int1.f90")
    ]

    def test_write_back(self):
        x = np.array(1, dtype=np.int8)
        self.module.inc_i1(x)
        assert x == 2


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


def test_meson_paths_relative_to_cwd(tmp_path, monkeypatch):
    # Command line paths are relative to the current directory, while the
    # meson.build file is written to, and resolved from, the build directory
    monkeypatch.chdir(tmp_path)
    build_dir = tmp_path / "bbuild"
    (build_dir / "inc").mkdir(parents=True)
    backend = MesonBackend(
        modulename="Blah",
        sources=[],
        extra_objects=["obj/extra.o"],
        build_dir=build_dir,
        include_dirs=["mods", build_dir / "inc"],
        library_dirs=["lib"],
        libraries=["foo"],
        define_macros=[],
        undef_macros=[],
        f2py_flags=[],
        sysinfo_flags=[],
        fc_flags=[],
        flib_flags=[],
        setup_flags=[],
        remove_build_dir=False,
        extra_dat={},
    )
    meson_build = backend.write_meson_build(build_dir).read_text()

    assert f"'''{Path('mods').resolve()}'''" in meson_build
    assert f"'''-L{Path('lib').absolute()}'''" in meson_build
    assert f"'''{Path('obj', 'extra.o').absolute()}'''" in meson_build
    # meson only accepts relative paths to directories in the build directory
    assert "'''inc'''" in meson_build


def test_meson_library_files_after_objects():
    # Libraries given as files must follow the objects that use them on the
    # link line, which meson does for link_args but not for objects
    meson_build = MesonTemplate(
        modulename="Blah",
        sources=[Path("f90continuation.f90")],
        deps=[],
        libraries=["bar"],
        library_dirs=[],
        include_dirs=[],
        object_files=[Path("/x/extra.o"), Path("/x/libfoo.a"), Path("/x/libbaz.so")],
        linker_args=[],
        fortran_args=[],
        build_type="debug",
        python_exe=sys.executable,
    ).generate_meson_build()

    objects = meson_build.split("objects: [")[1].split("]")[0]
    assert "'''/x/extra.o'''" in objects
    assert "libfoo.a" not in objects
    assert "libbaz.so" not in objects
    assert "declare_dependency(link_args : ['''/x/libfoo.a'''])" in meson_build
    assert "declare_dependency(link_args : ['''/x/libbaz.so'''])" in meson_build
    deps = meson_build.split("dependencies : [")[1]
    assert (deps.index("lib_file_0_libfoo_a") < deps.index("lib_file_1_libbaz_so")
            < deps.index("lib_0_bar"))


def test_gh25415_meson_include_dirs_from_running_numpy():
    # meson setup must not spawn Python to look up the include directories
    mparser = pytest.importorskip("mesonbuild.mparser")
    meson_build = MesonTemplate(
        modulename="Blah",
        sources=[Path("f90continuation.f90")],
        deps=[],
        libraries=[],
        library_dirs=[],
        include_dirs=[],
        object_files=[],
        linker_args=[],
        fortran_args=[],
        build_type="release",
        python_exe=sys.executable,
    ).generate_meson_build()

    mparser.Parser(meson_build, "meson.build").parse()
    assert "run_command" not in meson_build
    assert f"incdir_numpy = '''{np.get_include()}'''" in meson_build
    assert f"incdir_f2py = '''{np.f2py.get_include()}'''" in meson_build


def test_gh26352_intel_windows_symbol_names():
    # F_FUNC expects lowercase Fortran symbols with a trailing underscore
    mparser = pytest.importorskip("mesonbuild.mparser")
    meson_build = MesonTemplate(
        modulename="fib1",
        sources=[Path("fib1.f90")],
        deps=[],
        libraries=[],
        library_dirs=[],
        include_dirs=[],
        object_files=[],
        linker_args=[],
        fortran_args=[],
        build_type="release",
        python_exe=sys.executable,
    ).generate_meson_build()

    mparser.Parser(meson_build, "meson.build").parse()
    intel_windows = "if fc.get_id() in ['intel-cl', 'intel-llvm-cl']"
    assert intel_windows in meson_build
    start = meson_build.index(intel_windows)
    block = meson_build[start:meson_build.index("endif", start)]
    assert "'/names:lowercase', '/assume:underscore'" in block
    assert "language : 'fortran'" in block


@pytest.mark.slow
class TestPrecompiledModuleData(util.F2PyTest):
    # gh-19162: wrap the data of a module compiled into a library without
    # f2py, passing -I, -L and -l relative to the current directory
    module_name = "gh19162_ext"
    sources = [util.getpath("tests", "src", "regression", "gh19162", "gh19162.pyf")]

    @classmethod
    def setup_class(cls):
        super().setup_class()
        if not cls._has_fortran_compiler:
            pytest.skip("No Fortran compiler available")
        lib_dir, mod_dir = util.build_fortran_library(
            [util.getpath("tests", "src", "regression", "gh19162", "gh19162_lib.f90")],
            "gh19162",
        )
        module_dir = util.get_module_dir()
        cls.options = [
            f"-I{os.path.relpath(mod_dir, module_dir)}",
            f"-L{os.path.relpath(lib_dir, module_dir)}",
            "-lgh19162",
        ]

    def test_gh19162(self):
        mod = self.module.gh19162_mod
        mod.x = [1, 2, 3]
        assert mod.total() == 6


@pytest.mark.slow
class TestPrecompiledModuleUse(util.F2PyTest):
    # gh-19162: a module that uses one from a static library given as a file,
    # exposing a variable of the used module through an !f2py declaration
    sources = [util.getpath("tests", "src", "regression", "gh19162", "gh19162_use.f90")]

    @classmethod
    def setup_class(cls):
        super().setup_class()
        if not cls._has_fortran_compiler:
            pytest.skip("No Fortran compiler available")
        lib_dir, mod_dir = util.build_fortran_library(
            [util.getpath("tests", "src", "regression", "gh19162", "gh19162_lib.f90")],
            "gh19162",
        )
        cls.options = [f"-I{mod_dir}", str(lib_dir / "libgh19162.a")]

    def test_gh19162(self):
        mod = self.module.gh19162_use
        mod.x = [1, 2, 3]
        assert mod.scaled_total(2) == 12


@pytest.mark.slow
def test_gh25415_meson_configures_host_compilers_only(tmp_path):
    # Native builds never use the build machine compilers, so meson setup
    # should not detect and sanity check them a second time
    mod = util.build_code(
        """
        subroutine gh25415(x)
          real(8), intent(inout) :: x
          x = x + 1
        end subroutine
        """,
        options=["--build-dir", str(tmp_path)],
        suffix=".f90",
    )
    x = np.array(1.0)
    mod.gh25415(x)
    assert x == 2.0

    intro, = tmp_path.glob("*/meson-info/intro-compilers.json")
    compilers = json.loads(intro.read_text())
    assert set(compilers["host"]) == {"c", "fortran"}
    assert not compilers.get("build")
    intel_windows = "if fc.get_id() in ['intel-cl', 'intel-llvm-cl']"
    assert intel_windows in meson_build
    start = meson_build.index(intel_windows)
    block = meson_build[start:meson_build.index("endif", start)]
    assert "'/names:lowercase', '/assume:underscore'" in block
    assert "language : 'fortran'" in block


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
def test_gh25447(monkeypatch, tmp_path):
    # Compiler checks must survive a probe directory that cannot be removed,
    # as on Windows while another process still holds meson.build open
    unlink = os.unlink
    busy = []

    def busy_unlink(path, *args, **kwargs):
        if os.path.basename(path) == "meson.build":
            busy.append(path)
            raise PermissionError(errno.EACCES, "File in use", path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(os, "unlink", busy_unlink)
    checker = util.CompilerChecker()
    checker.check_compilers()

    assert busy
    assert (checker.has_c, checker.has_f77, checker.has_f90) == (
        util.has_c_compiler(), util.has_f77_compiler(), util.has_f90_compiler()
    )


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


def test_gh30167_callstatement_prototype(tmp_path):
    # Without callprotoargument the prototype of f2py_func must come from
    # the call in callstatement: C23 reads an empty parameter list as (void)
    pyf = util.getpath("tests", "src", "regression", "gh30167.pyf")
    # A subprocess, as f2py leaves crackfortran.dolowercase set afterwards.
    # Its working directory is the build directory so Python does not put a
    # source checkout on sys.path ahead of the installed package.
    subprocess.run([sys.executable, "-m", "numpy.f2py", str(pyf),
                    "--build-dir", str(tmp_path)],
                   check=True, capture_output=True, cwd=tmp_path)
    cmod = (tmp_path / "_gh30167module.c").read_text()
    assert "void (*f2py_func)())" not in cmod
    assert "F_FUNC(scale,SCALE)(int*,double*,double*);" in cmod
    assert ("F_FUNC(apply,APPLY)(cb_fun_in__gh30167__user__routines_typedef,"
            "int*,double*);") in cmod


@pytest.mark.slow
class TestCallstatementPrototype(util.F2PyTest):
    # gh-30167: callstatement without callprotoargument, passing the
    # arguments in another order than the signature and a call-back
    sources = [
        util.getpath("tests", "src", "regression", "gh30167.pyf"),
        util.getpath("tests", "src", "regression", "gh30167.f"),
    ]
    module_name = "_gh30167"

    def test_reordered_arguments(self):
        x = np.arange(1.0, 4.0)
        npt.assert_array_equal(self.module.scale(x, 2.0), [2.0, 4.0, 6.0])

    def test_callback_argument(self):
        x = np.arange(1.0, 4.0)
        npt.assert_array_equal(self.module.apply(lambda v: v * v, x),
                               [1.0, 4.0, 9.0])


@pytest.mark.slow
class TestSharedLibraryInterface(util.F2PyTest):
    # Check that explicit interfaces to procedures in shared libraries link
    # with compilers that apply -fvisibility to references, like ifx
    sources = [util.getpath("tests", "src", "regression", "shlib_interface.f90")]

    def test_gh27163(self):
        assert self.module.get_pid() == os.getpid()


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


@pytest.mark.slow
class TestContainedProcedures(util.F2PyTest):
    # Check that CONTAINS procedures stay out of the F90 wrappers
    sources = [util.getpath("tests", "src", "regression", "gh20103.f90")]

    def test_gh20103_subroutine(self):
        array = np.zeros((3, 3), order="F")
        self.module.example(array)
        expected = 100 * np.arange(1, 4)[:, None] + 10 * np.arange(1, 4)
        npt.assert_array_equal(array, expected)

    def test_gh20103_function(self):
        assert self.module.total([1.0, 2.0, 3.0]) == 12.0

    def test_gh20103_module_procedure(self):
        x = np.ones(3)
        self.module.gh20103_mod.scale(x)
        npt.assert_array_equal(x, 2.0)
class TestInOutInt1(util.F2PyTest):
    # gh-25229: intent(inout) signed_char scalars need pyobj_from_signed_char1
    sources = [util.getpath("tests", "src", "regression", "inout_int1.f90")]

    @pytest.mark.parametrize("name", ["inc_int1", "inc_c_int8"])
    @pytest.mark.parametrize("dtype", [np.int8, object])
    def test_inout(self, name, dtype):
        x = np.array(41, dtype=dtype)
        getattr(self.module, name)(x)
        assert x == 42


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


@pytest.mark.slow
class TestPickleFortranRoutine(util.F2PyTest):
    # gh-21767: routine objects pickle by reference. Data objects refuse.
    sources = [util.getpath("tests", "src", "regression", "gh21767.f90")]

    def test_attribute_identity(self):
        assert self.module.s is self.module.s
        assert self.module.m.t is self.module.m.t

    def test_module_and_qualname(self):
        mod = self.module
        assert mod.s.__module__ == mod.__name__
        assert mod.s.__qualname__ == "s"
        assert mod.m.t.__module__ == mod.__name__
        assert mod.m.t.__qualname__ == "m.t"

    def test_pickle_protocols(self):
        highest = pickle.HIGHEST_PROTOCOL
        routines = (self.module.s, self.module.m.t)
        for proto in range(highest + 1):
            for routine in routines:
                blob = pickle.dumps(routine, protocol=proto)
                assert pickle.loads(blob) is routine

    def test_copy_identity(self):
        routines = (self.module.s, self.module.m.t)
        for routine in routines:
            assert copy.copy(routine) is routine
            assert copy.deepcopy(routine) is routine

    def test_cloudpickle_roundtrip(self):
        cloudpickle = pytest.importorskip("cloudpickle")
        routines = (self.module.s, self.module.m.t)
        for routine in routines:
            blob = cloudpickle.dumps(routine)
            assert cloudpickle.loads(blob) is routine

    def _assert_pickle_label_hidden(self, obj):
        assert "_f2py_pickle_kind" not in obj.__dict__
        assert "_f2py_pickle_kind" not in dir(obj)
        assert "_f2py_pickle_name" not in obj.__dict__
        assert "_f2py_pickle_name" not in dir(obj)

    def test_f90_module_refuses_pickle(self):
        obj = self.module.m
        self._assert_pickle_label_hidden(obj)
        with pytest.raises(
            TypeError,
            match=(
                "F90 module 'm': f2py data objects wrap Fortran memory"
                " and cannot be pickled; copy the array data instead"
            ),
        ):
            pickle.dumps(obj)

    def test_variable_container_refuses_pickle(self):
        obj = self.module.vmod
        self._assert_pickle_label_hidden(obj)
        with pytest.raises(
            TypeError,
            match=(
                "module variable container 'vmod': f2py data objects wrap"
                " Fortran memory and cannot be pickled; copy the array"
                " data instead"
            ),
        ):
            pickle.dumps(obj)

    def test_common_refuses_pickle(self):
        obj = self.module.cblock
        self._assert_pickle_label_hidden(obj)
        with pytest.raises(
            TypeError,
            match=(
                "COMMON block 'cblock': f2py data objects wrap Fortran"
                " memory and cannot be pickled; copy the array data"
                " instead"
            ),
        ):
            pickle.dumps(obj)
# Fortran helpers multiply by this factor. See gh24394.f90.
_GH24394_FACTOR = 2
# Value that stays exact through both the integer and real converters.
_GH24394_SCALAR = 2
# Second element. It must differ from the scalar so a wrong pick is visible.
_GH24394_IGNORED = 9
# Nonzero imaginary part. A zero part is the quiet complex case.
_GH24394_IMAG = 3
# 5/2 truncates to 2, so an integer converter drops a fractional part.
_GH24394_FRACTION_NUM = 5
_GH24394_FRACTION_DEN = 2

_GH24394_ROUTINES = ("double_it", "double_int", "double_i8")
_GH24394_INT_ROUTINES = ("double_int", "double_i8")

_GH24394_COMPLEX_WARNING = (
    r"f2py scalar argument got a complex value with a nonzero "
    r"imaginary part; dropping it is deprecated \(gh-24394\)"
)
_GH24394_TRUNC_WARNING = (
    r"f2py scalar argument got a non-integer float for an integer "
    r"argument; truncating it is deprecated \(gh-24394\)"
)


def _gh24394_sequence_warning(length):
    return (
        rf"f2py scalar argument got a sequence of length {length}; "
        r"only the first element is used, which is deprecated \(gh-24394\)"
    )


def _gh24394_number(routine, value):
    if routine == "double_it":
        return float(value)
    return value


def _gh24394_fraction():
    return _GH24394_FRACTION_NUM / _GH24394_FRACTION_DEN


@pytest.mark.slow
class TestScalarConversionDeprecation(util.F2PyTest):
    sources = [util.getpath("tests", "src", "regression", "gh24394.f90")]

    def _call(self, routine):
        return getattr(self.module, routine)

    def _expected(self, routine):
        return _gh24394_number(routine, _GH24394_SCALAR) * _GH24394_FACTOR

    def _quiet_values(self, routine):
        scalar = _gh24394_number(routine, _GH24394_SCALAR)
        return (
            scalar,
            complex(scalar, 0),
            [scalar],
            np.array([scalar]),
            [[scalar]],
        )

    def _lossy_sequences(self, routine):
        scalar = _gh24394_number(routine, _GH24394_SCALAR)
        ignored = _gh24394_number(routine, _GH24394_IGNORED)
        pair = [scalar, ignored]
        return (pair, (scalar, ignored), np.array(pair), [pair])

    @pytest.mark.parametrize("routine", _GH24394_ROUTINES)
    def test_quiet_values(self, routine):
        fn = self._call(routine)
        expected = self._expected(routine)
        for value in self._quiet_values(routine):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = fn(value)
            assert result == expected

    @pytest.mark.parametrize("routine", _GH24394_ROUTINES)
    def test_nonzero_imaginary_part(self, routine):
        fn = self._call(routine)
        value = complex(
            _gh24394_number(routine, _GH24394_SCALAR),
            _GH24394_IMAG,
        )
        with pytest.warns(DeprecationWarning, match=_GH24394_COMPLEX_WARNING):
            result = fn(value)
        assert result == self._expected(routine)

    @pytest.mark.parametrize("routine", _GH24394_ROUTINES)
    def test_long_sequences(self, routine):
        fn = self._call(routine)
        expected = self._expected(routine)
        sequences = self._lossy_sequences(routine)
        # Every lossy row warns about the inner pair, whose length is two.
        warned = _gh24394_sequence_warning(len(sequences[0]))
        for value in sequences:
            with pytest.warns(DeprecationWarning, match=warned):
                result = fn(value)
            assert result == expected

    @pytest.mark.parametrize("routine", _GH24394_ROUTINES)
    def test_long_sequence_as_error(self, routine):
        fn = self._call(routine)
        scalar = _gh24394_number(routine, _GH24394_SCALAR)
        ignored = _gh24394_number(routine, _GH24394_IGNORED)
        warned = _gh24394_sequence_warning(len((scalar, ignored)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DeprecationWarning, match=warned):
                fn([scalar, ignored])

    @pytest.mark.parametrize("routine", _GH24394_ROUTINES)
    def test_empty_sequence(self, routine):
        fn = self._call(routine)
        with pytest.raises(IndexError):
            fn([])

    @pytest.mark.parametrize("routine", _GH24394_INT_ROUTINES)
    def test_integral_float_is_quiet(self, routine):
        fn = self._call(routine)
        value = float(_GH24394_SCALAR)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = fn(value)
        assert result == self._expected(routine)

    @pytest.mark.parametrize("routine", _GH24394_INT_ROUTINES)
    def test_nonintegral_float(self, routine):
        fn = self._call(routine)
        value = _gh24394_fraction()
        expected = self._expected(routine)
        cases = (
            value,
            np.float64(value),
            [value],
            complex(value, 0),
        )
        for item in cases:
            with pytest.warns(
                DeprecationWarning, match=_GH24394_TRUNC_WARNING
            ) as recorded:
                result = fn(item)
            assert len(recorded) == 1
            assert result == expected

    @pytest.mark.parametrize("routine", _GH24394_INT_ROUTINES)
    def test_nonintegral_float_as_error(self, routine):
        fn = self._call(routine)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                DeprecationWarning, match=_GH24394_TRUNC_WARNING
            ):
                fn(_gh24394_fraction())

    def test_real_argument_keeps_fraction(self):
        fn = self._call("double_it")
        value = _gh24394_fraction()
        expected = value * _GH24394_FACTOR
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = fn(value)
        assert result == expected

    @pytest.mark.parametrize("routine", _GH24394_INT_ROUTINES)
    @pytest.mark.parametrize("value", (float("nan"), float("inf")))
    def test_nonfinite_float_warns(self, routine, value):
        fn = self._call(routine)
        with pytest.warns(DeprecationWarning, match=_GH24394_TRUNC_WARNING):
            with pytest.raises((ValueError, OverflowError)):
                fn(value)
@pytest.mark.slow
def test_gh25159_mpi_compiler_wrapper(monkeypatch):
    wrapper = (os.environ.get("MPIFC") or os.environ.get("MPIF90")
               or shutil.which("mpifort") or shutil.which("mpif90"))
    if wrapper is None:
        pytest.skip("No MPI Fortran compiler wrapper available")
    monkeypatch.setenv("FC", wrapper)
    module = util.build_module(
        [util.getpath("tests", "src", "regression", "gh25159.f90")],
    )
    (major, minor, expected_major, expected_minor, initialized,
     initialized_status, version_status, success) = module.mpi_wrapper_info()
    assert (major, minor) == (expected_major, expected_minor)
    assert not initialized
    assert initialized_status == success
    assert version_status == success

def _meson_build_with_deps(deps):
    return MesonTemplate(
        modulename="Blah",
        sources=[Path("gh28902.f90")],
        deps=deps,
        libraries=[],
        library_dirs=[],
        include_dirs=[],
        object_files=[],
        linker_args=[],
        fortran_args=[],
        build_type="debug",
        python_exe=sys.executable,
    ).generate_meson_build()


def test_gh28902_dependency_keywords():
    mparser = pytest.importorskip("mesonbuild.mparser")
    meson_build = _meson_build_with_deps([
        "lapack",
        "mpi[language: 'fortran']",
        "boost [modules: ['thread', 'system'], static: true]",
    ])

    mparser.Parser(meson_build, "meson.build").parse()
    assert "dependency('lapack')," in meson_build
    assert "dependency('mpi', language: 'fortran')," in meson_build
    assert ("dependency('boost', modules: ['thread', 'system'], static: true),"
            in meson_build)


@pytest.mark.parametrize("dep", ["mpi[language: 'fortran'", "[static: true]"])
def test_gh28902_invalid_dependency(dep):
    with pytest.raises(ValueError, match="Invalid --dep value"):
        _meson_build_with_deps([dep])


@pytest.mark.slow
def test_gh28902_fortran_mpi():
    # Plain --dep mpi gets meson's C MPI, which lacks the Fortran bindings,
    # so the import fails with an undefined mpi_initialized_
    try:
        module = util.build_module(
            [util.getpath("tests", "src", "regression", "gh28902.f90")],
            options=["--dep", "mpi[language: 'fortran']"],
        )
    except RuntimeError as rerr:
        if 'Dependency "mpi" not found' in str(rerr):
            pytest.skip("No MPI for Fortran found by meson")
        raise
    assert not module.mpi_initialized_flag()


@pytest.mark.slow
@pytest.mark.skipif(platform.system() == "Windows", reason='Unsupported on this platform for now')
