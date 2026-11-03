import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from . import util


def get_docdir():
    parents = Path(__file__).resolve().parents
    try:
        # Assumes that spin is used to run tests
        nproot = parents[8]
    except IndexError:
        docdir = None
    else:
        docdir = nproot / "doc" / "source" / "f2py" / "code"
    if docdir and docdir.is_dir():
        return docdir
    # Assumes that an editable install is used to run tests
    return parents[3] / "doc" / "source" / "f2py" / "code"


pytestmark = pytest.mark.skipif(
    not get_docdir().is_dir(),
    reason=f"Could not find f2py documentation sources"
    f"({get_docdir()} does not exist)",
)

def _path(*args):
    return get_docdir().joinpath(*args)

@pytest.mark.slow
class TestDocAdvanced(util.F2PyTest):
    # options = ['--debug-capi', '--build-dir', '/tmp/build-f2py']
    sources = [_path('asterisk1.f90'), _path('asterisk2.f90'),
               _path('ftype.f')]

    def test_asterisk1(self):
        foo = self.module.foo1
        assert_equal(foo(), b'123456789A12')

    def test_asterisk2(self):
        foo = self.module.foo2
        assert_equal(foo(2), b'12')
        assert_equal(foo(12), b'123456789A12')
        assert_equal(foo(20), b'123456789A123456789B')

    def test_ftype(self):
        ftype = self.module
        ftype.foo()
        assert_equal(ftype.data.a, 0)
        ftype.data.a = 3
        ftype.data.x = [1, 2, 3]
        assert_equal(ftype.data.a, 3)
        assert_array_equal(ftype.data.x,
                           np.array([1, 2, 3], dtype=np.float32))
        ftype.data.x[1] = 45
        assert_array_equal(ftype.data.x,
                           np.array([1, 45, 3], dtype=np.float32))
        # gh-26718 Cleanup for repeated test runs
        ftype.data.a = 0

    # TODO: implement test methods for other example Fortran codes


def _meson_build(project):
    # The examples build for the Python that runs meson unless told otherwise
    native = project / "native.ini"
    native.write_text(f"[binaries]\npython = '{Path(sys.executable).as_posix()}'\n")
    for cmd in (["meson", "setup", "builddir", "--native-file", native.name],
                ["meson", "compile", "-C", "builddir"]):
        run = subprocess.run(cmd, cwd=project, capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
    return project / "builddir"


def _run_built(builddir, code):
    # Not imported here: the examples' modules are not declared free-threading
    # compatible, so importing one re-enables the GIL with a RuntimeWarning
    run = subprocess.run([sys.executable, "-c", code], cwd=builddir,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.slow
@pytest.mark.parametrize("meson_build, module",
                         [("meson.build", "fib2"), ("meson_upd.build", "fibby")])
def test_gh26973_meson_example(meson_build, module, tmp_path, monkeypatch):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    project = tmp_path / "project"
    util.numpy_inside(project, monkeypatch)
    shutil.copy(_path("fib1.f"), project)
    shutil.copy(_path(meson_build), project / "meson.build")
    # meson.build compiles the wrapper that the walkthrough generates first
    subprocess.run([sys.executable, "-m", "numpy.f2py", "fib1.f", "-m", "fib2",
                    "--lower"], cwd=project, check=True, capture_output=True)
    _run_built(_meson_build(project),
               f"import numpy as np, {module}\n"
               "a = np.zeros(9)\n"
               f"{module}.fib(a)\n"
               "assert a.tolist() == [0, 1, 1, 2, 3, 5, 8, 13, 21], a\n")


@pytest.mark.slow
def test_gh26973_meson_python_example(tmp_path, monkeypatch):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    if shutil.which("pkg-config") is None:
        pytest.skip("dependency('numpy') is found through pkg-config here")
    project = tmp_path / "project"
    util.numpy_inside(project, monkeypatch)
    # What ``numpy-config --pkgconfigdir`` prints, ahead of any other numpy.pc
    monkeypatch.setenv("PKG_CONFIG_PATH",
                       str(Path(np.get_include()).parent / "lib" / "pkgconfig"))
    shutil.copy(_path("fib_mesonpy.f90"), project / "fib.f90")
    shutil.copy(_path("meson_mesonpy.build"), project / "meson.build")
    (project / "fib_wrapper").mkdir()
    (project / "fib_wrapper" / "__init__.py").write_text("from .fib import fib\n")
    _run_built(_meson_build(project),
               "import fib\n"
               "a = fib.fib(10)\n"
               "assert a.tolist() == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34], a\n")
