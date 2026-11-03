"""
Utility functions for

- building and importing modules on test time, using a temporary location
- detecting if compilers are present
- determining paths to tests

"""
import atexit
import concurrent.futures
import contextlib
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import pytest

import numpy
from numpy._utils import asunicode
from numpy.f2py._backends._meson import MesonBackend
from numpy.testing import IS_IOS, IS_WASM, temppath

#
# Check if compilers are available at all...
#

def check_language(lang, code_snippet=None):
    if sys.platform == "win32":
        pytest.skip("No Fortran tests on Windows (Issue #25134)", allow_module_level=True)
    tmpdir = tempfile.mkdtemp()
    try:
        meson_file = os.path.join(tmpdir, "meson.build")
        with open(meson_file, "w") as f:
            f.write("project('check_compilers')\n")
            f.write(f"add_languages('{lang}')\n")
            if code_snippet:
                f.write(f"{lang}_compiler = meson.get_compiler('{lang}')\n")
                f.write(f"{lang}_code = '''{code_snippet}'''\n")
                f.write(
                    f"_have_{lang}_feature ="
                    f"{lang}_compiler.compiles({lang}_code,"
                    f" name: '{lang} feature check')\n"
                )
        try:
            runmeson = subprocess.run(
                ["meson", "setup", "btmp"],
                check=False,
                cwd=tmpdir,
                capture_output=True,
            )
        except OSError:
            pytest.skip("meson not present, skipping compiler dependent test", allow_module_level=True)
        return runmeson.returncode == 0
    finally:
        shutil.rmtree(tmpdir)


fortran77_code = '''
C Example Fortran 77 code
      PROGRAM HELLO
      PRINT *, 'Hello, Fortran 77!'
      END
'''

fortran90_code = '''
! Example Fortran 90 code
program hello90
  type :: greeting
    character(len=20) :: text
  end type greeting

  type(greeting) :: greet
  greet%text = 'hello, fortran 90!'
  print *, greet%text
end program hello90
'''

# Dummy class for caching relevant checks
class CompilerChecker:
    def __init__(self):
        self.compilers_checked = False
        self.has_c = False
        self.has_f77 = False
        self.has_f90 = False

    def check_compilers(self):
        if (not self.compilers_checked):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(check_language, "c"),
                    executor.submit(check_language, "fortran", fortran77_code),
                    executor.submit(check_language, "fortran", fortran90_code)
                ]

                self.has_c = futures[0].result()
                self.has_f77 = futures[1].result()
                self.has_f90 = futures[2].result()

            self.compilers_checked = True


if not IS_WASM and not IS_IOS:
    checker = CompilerChecker()
    checker.check_compilers()

def has_c_compiler():
    return checker.has_c

def has_f77_compiler():
    return checker.has_f77

def has_f90_compiler():
    return checker.has_f90

def has_fortran_compiler():
    return (checker.has_f90 and checker.has_f77)


def fortran_builds(code, args=()):
    """Return whether the Fortran compiler builds ``code`` into an object,
    with the compiler arguments ``args``.

    ``compiler.compiles()`` cannot answer this: its result is not checked by
    ``check_language``, and for LLVM flang Meson adds clang-only
    ``-Werror=`` options that make every check fail.  Building a static
    library uses the same flags as the extension modules under test.
    """
    if sys.platform == "win32" or not has_f90_compiler():
        return False
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "meson.build"), "w") as f:
            f.write("project('check_build', 'fortran')\n")
            f.write(f"static_library('check', 'check.f90', "
                    f"fortran_args: {list(args)!r})\n")
        with open(os.path.join(tmpdir, "check.f90"), "w") as f:
            f.write(code)
        for cmd in (["meson", "setup", "btmp"],
                    ["meson", "compile", "-C", "btmp"]):
            if subprocess.run(cmd, check=False, cwd=tmpdir,
                              capture_output=True).returncode != 0:
                return False
        return True
    finally:
        shutil.rmtree(tmpdir)


_pdt_support = None


def has_pdt_support():
    """Check if the Fortran compiler supports KIND type parameters.

    Cached after the first call.
    """
    global _pdt_support
    if _pdt_support is None:
        _pdt_support = fortran_builds(
            'module pdt_check\n'
            '  implicit none\n'
            '  type :: vec(k)\n'
            '    integer, kind :: k = kind(0.0d0)\n'
            '    real(k) :: x\n'
            '  end type vec\n'
            'contains\n'
            '  subroutine make(v)\n'
            '    type(vec(kind(0.0))), intent(out) :: v\n'
            '    v%x = 0\n'
            '  end subroutine make\n'
            'end module pdt_check\n'
        )
    return _pdt_support


_pdt_len_support = None


def has_pdt_len_support():
    """Check if the Fortran compiler handles LEN type parameters as f2py
    wraps them.

    The handle of a LEN-parameterized object is ``c_loc`` of a pointer
    allocated with the requested length.  F2018 18.2.3.6 only allows
    ``c_loc`` on noninteroperable scalars without length type parameters,
    so this needs a compiler that accepts it as an extension; LLVM flang
    rejects it and cannot lower LEN-parameterized objects either.  Cached
    after the first call.
    """
    global _pdt_len_support
    if _pdt_len_support is None:
        _pdt_len_support = fortran_builds(
            'module pdt_len_check\n'
            '  use iso_c_binding, only: c_ptr, c_loc\n'
            '  implicit none\n'
            '  type :: vec(n)\n'
            '    integer, len :: n\n'
            '    real :: x(n)\n'
            '  end type vec\n'
            'contains\n'
            '  function make(n) result(cptr)\n'
            '    integer, intent(in) :: n\n'
            '    type(c_ptr) :: cptr\n'
            '    type(vec(n=n)), pointer :: v\n'
            '    allocate(vec(n=n) :: v)\n'
            '    cptr = c_loc(v)\n'
            '  end function make\n'
            'end module pdt_len_check\n'
        )
    return _pdt_len_support


#
# Maintaining a temporary module directory
#

_module_dir = None
_module_num = 5403

if sys.platform == "cygwin":
    NUMPY_INSTALL_ROOT = Path(__file__).parent.parent.parent
    _module_list = list(NUMPY_INSTALL_ROOT.glob("**/*.dll"))


def _cleanup():
    global _module_dir
    if _module_dir is not None:
        try:
            sys.path.remove(_module_dir)
        except ValueError:
            pass
        try:
            shutil.rmtree(_module_dir)
        except OSError:
            pass
        _module_dir = None


def get_module_dir():
    global _module_dir
    if _module_dir is None:
        _module_dir = tempfile.mkdtemp()
        atexit.register(_cleanup)
        if _module_dir not in sys.path:
            sys.path.insert(0, _module_dir)
    return _module_dir


def get_temp_module_name():
    # Assume single-threaded, and the module dir usable only by this thread
    global _module_num
    get_module_dir()
    name = f"_test_ext_module_{_module_num}"
    _module_num += 1
    if name in sys.modules:
        # this should not be possible, but check anyway
        raise RuntimeError("Temporary module name already in use.")
    return name


def _memoize(func):
    memo = {}

    def wrapper(*a, **kw):
        key = repr((a, kw))
        if key not in memo:
            try:
                memo[key] = func(*a, **kw)
            except Exception as e:
                memo[key] = e
                raise
        ret = memo[key]
        if isinstance(ret, Exception):
            raise ret
        return ret

    wrapper.__name__ = func.__name__
    return wrapper


#
# Building modules
#


@_memoize
def build_module(source_files, options=[], skip=[], only=[], module_name=None):
    """
    Compile and import a f2py module, built from the given files.

    """

    code = f"import sys; sys.path = {sys.path!r}; import numpy.f2py; numpy.f2py.main()"

    d = get_module_dir()
    # gh-27045 : Skip if no compilers are found
    if not has_fortran_compiler():
        pytest.skip("No Fortran compiler available")

    # Copy files
    dst_sources = []
    f2py_sources = []
    for fn in source_files:
        if not os.path.isfile(fn):
            raise RuntimeError(f"{fn} is not a file")
        dst = os.path.join(d, os.path.basename(fn))
        shutil.copyfile(fn, dst)
        dst_sources.append(dst)

        base, ext = os.path.splitext(dst)
        if ext in (".f90", ".f95", ".f", ".c", ".pyf"):
            f2py_sources.append(dst)

    assert f2py_sources

    # Prepare options
    if module_name is None:
        module_name = get_temp_module_name()
    gil_options = []
    if '--freethreading-compatible' not in options and '--no-freethreading-compatible' not in options:
        # default to disabling the GIL if unset in options
        gil_options = ['--freethreading-compatible']
    f2py_opts = ["-c", "-m", module_name] + options + gil_options + f2py_sources
    f2py_opts += ["--backend", "meson"]
    if skip:
        f2py_opts += ["skip:"] + skip
    if only:
        f2py_opts += ["only:"] + only

    # Build
    cwd = os.getcwd()
    try:
        os.chdir(d)
        cmd = [sys.executable, "-c", code] + f2py_opts
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"Running f2py failed: {cmd[4:]}\n{asunicode(out)}")
    finally:
        os.chdir(cwd)

        # Partial cleanup
        for fn in dst_sources:
            os.unlink(fn)

    # Rebase (Cygwin-only)
    if sys.platform == "cygwin":
        # If someone starts deleting modules after import, this will
        # need to change to record how big each module is, rather than
        # relying on rebase being able to find that from the files.
        _module_list.extend(
            glob.glob(os.path.join(d, f"{module_name:s}*"))
        )
        subprocess.check_call(
            ["/usr/bin/rebase", "--database", "--oblivious", "--verbose"]
            + _module_list
        )

    # Import
    return import_module(module_name)


@_memoize
def build_code(source_code,
               options=[],
               skip=[],
               only=[],
               suffix=None,
               module_name=None):
    """
    Compile and import Fortran code using f2py.

    """
    if suffix is None:
        suffix = ".f"
    with temppath(suffix=suffix) as path:
        with open(path, "w") as f:
            f.write(source_code)
        return build_module([path],
                            options=options,
                            skip=skip,
                            only=only,
                            module_name=module_name)


#
# Building with meson
#


class SimplifiedMesonBackend(MesonBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compile(self):
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)


def build_meson(source_files, module_name=None, **kwargs):
    """
    Build a module via Meson and import it.
    """

    # gh-27045 : Skip if no compilers are found
    if not has_fortran_compiler():
        pytest.skip("No Fortran compiler available")

    build_dir = get_module_dir()
    if module_name is None:
        module_name = get_temp_module_name()

    # Initialize the MesonBackend
    backend = SimplifiedMesonBackend(
        modulename=module_name,
        sources=source_files,
        extra_objects=kwargs.get("extra_objects", []),
        build_dir=build_dir,
        include_dirs=kwargs.get("include_dirs", []),
        library_dirs=kwargs.get("library_dirs", []),
        libraries=kwargs.get("libraries", []),
        define_macros=kwargs.get("define_macros", []),
        undef_macros=kwargs.get("undef_macros", []),
        f2py_flags=kwargs.get("f2py_flags", []),
        sysinfo_flags=kwargs.get("sysinfo_flags", []),
        fc_flags=kwargs.get("fc_flags", []),
        flib_flags=kwargs.get("flib_flags", []),
        setup_flags=kwargs.get("setup_flags", []),
        remove_build_dir=kwargs.get("remove_build_dir", False),
        extra_dat=kwargs.get("extra_dat", {}),
    )

    backend.compile()

    # Import the compiled module
    sys.path.insert(0, f"{build_dir}/{backend.meson_build_dir}")
    return import_module(module_name)


#
# Unittest convenience
#


class F2PyTest:
    code = None
    sources = None
    options = []  # noqa: RUF012
    skip = []  # noqa: RUF012
    only = []  # noqa: RUF012
    suffix = ".f"
    module = None
    _has_c_compiler = None
    _has_f77_compiler = None
    _has_f90_compiler = None

    @property
    def module_name(self):
        cls = type(self)
        return f'_{cls.__module__.rsplit(".", 1)[-1]}_{cls.__name__}_ext_module'

    @classmethod
    def setup_class(cls):
        if sys.platform == "win32":
            pytest.skip("Fails with MinGW64 Gfortran (Issue #9673)")
        F2PyTest._has_c_compiler = has_c_compiler()
        F2PyTest._has_f77_compiler = has_f77_compiler()
        F2PyTest._has_f90_compiler = has_f90_compiler()
        F2PyTest._has_fortran_compiler = has_fortran_compiler()

    def setup_method(self):
        if self.module is not None:
            return

        codes = self.sources or []
        if self.code:
            codes.append(self.suffix)

        needs_f77 = any(str(fn).endswith(".f") for fn in codes)
        needs_f90 = any(str(fn).endswith(".f90") for fn in codes)
        needs_pyf = any(str(fn).endswith(".pyf") for fn in codes)

        if needs_f77 and not self._has_f77_compiler:
            pytest.skip("No Fortran 77 compiler available")
        if needs_f90 and not self._has_f90_compiler:
            pytest.skip("No Fortran 90 compiler available")
        if needs_pyf and not self._has_fortran_compiler:
            pytest.skip("No Fortran compiler available")

        # Build the module
        if self.code is not None:
            self.module = build_code(
                self.code,
                options=self.options,
                skip=self.skip,
                only=self.only,
                suffix=self.suffix,
                module_name=self.module_name,
            )

        if self.sources is not None:
            self.module = build_module(
                self.sources,
                options=self.options,
                skip=self.skip,
                only=self.only,
                module_name=self.module_name,
            )


#
# Helper functions
#


def getpath(*a):
    # Package root
    d = Path(numpy.f2py.__file__).parent.resolve()
    return d.joinpath(*a)


@contextlib.contextmanager
def switchdir(path):
    curpath = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curpath)
