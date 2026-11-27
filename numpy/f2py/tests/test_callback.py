import math
import platform
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import traceback

import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestF77Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "foo.f")]

    @pytest.mark.parametrize("name", ["t", "t2"])
    def test_all(self, name):
        self.check_function(name)

    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = t(fun,[fun_extra_args])

        Wrapper for ``t``.

        Parameters
        ----------
        fun : call-back function

        Other Parameters
        ----------------
        fun_extra_args : input tuple, optional
            Default: ()

        Returns
        -------
        a : int

        Notes
        -----
        Call-back functions::

            def fun(): return a
            Return objects:
                a : int
        """)
        assert self.module.t.__doc__ == expected

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda: 4)
        assert r == 4
        r = t(lambda a: 5, fun_extra_args=(6, ))
        assert r == 5
        r = t(lambda a: a, fun_extra_args=(6, ))
        assert r == 6
        r = t(lambda a: 5 + a, fun_extra_args=(7, ))
        assert r == 12
        r = t(math.degrees, fun_extra_args=(math.pi, ))
        assert r == 180
        r = t(math.degrees, fun_extra_args=(math.pi, ))
        assert r == 180

        r = t(self.module.func, fun_extra_args=(6, ))
        assert r == 17
        r = t(self.module.func0)
        assert r == 11
        r = t(self.module.func0._cpointer)
        assert r == 11

        class A:
            def __call__(self):
                return 7

            def mth(self):
                return 9

        a = A()
        r = t(a)
        assert r == 7
        r = t(a.mth)
        assert r == 9

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):
        def callback(code):
            if code == "r":
                return 0
            else:
                return 1

        f = self.module.string_callback
        r = f(callback)
        assert r == 0

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        # See gh-10027
        cu1 = np.zeros((1, ), "S8")
        cu2 = np.zeros((1, 8), "c")
        cu3 = np.array([""], "S8")

        def callback(cu, lencu):
            if cu.shape != (lencu,):
                return 1
            if cu.dtype != "S8":
                return 2
            if not np.all(cu == b""):
                return 3
            return 0

        f = self.module.string_callback_array
        for cu in [cu1, cu2, cu3]:
            res = f(callback, cu, cu.size)
            assert res == 0

    def test_threadsafety(self):
        # Segfaults if the callback handling is not threadsafe

        errors = []

        def cb():
            # Sleep here to make it more likely for another thread
            # to call their callback at the same time.
            time.sleep(1e-3)

            # Check reentrancy
            r = self.module.t(lambda: 123)
            assert r == 123

            return 42

        def runner(name):
            try:
                for j in range(50):
                    r = self.module.t(cb)
                    assert r == 42
                    self.check_function(name)
            except Exception:
                errors.append(traceback.format_exc())

        threads = [
            threading.Thread(target=runner, args=(arg, ))
            for arg in ("t", "t2") for n in range(20)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        errors = "\n\n".join(errors)
        if errors:
            raise AssertionError(errors)

    def test_hidden_callback(self):
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        try:
            self.module.hidden_callback2(2)
        except Exception as msg:
            assert str(msg).startswith("cb: Callback global_f not defined")

        self.module.global_f = lambda x: x + 1
        r = self.module.hidden_callback(2)
        assert r == 3

        self.module.global_f = lambda x: x + 2
        r = self.module.hidden_callback(2)
        assert r == 4

        del self.module.global_f
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        self.module.global_f = lambda x=0: x + 3
        r = self.module.hidden_callback(2)
        assert r == 5

        # reproducer of gh18341
        r = self.module.hidden_callback2(2)
        assert r == 3


@pytest.mark.slow
class TestF77CallbackPythonTLS(TestF77Callback):
    """
    Callback tests using Python thread-local storage instead of
    compiler-provided
    """

    options = ["-DF2PY_USE_PYTHON_TLS"]


@pytest.mark.slow
class TestF90Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh17797.f90")]

    def test_gh17797(self):
        def incr(x):
            return x + 123

        y = np.array([1, 2, 3], dtype=np.int64)
        r = self.module.gh17797(incr, y)
        assert r == 123 + 1 + 2 + 3


def _gh20157_r_decls(text):
    """Split ``:: r`` lines into callback results and host dummies."""
    callback, host = [], []
    in_fun = False
    for line in text.splitlines():
        if (not in_fun and "function " in line and "result (r)" in line
                and "e_0_e" in line):
            in_fun = True
            continue
        if in_fun and "end function" in line:
            in_fun = False
            continue
        if ":: r" not in line:
            continue
        (callback if in_fun else host).append(line.strip())
    return callback, host


def test_gh20157_callback_result(tmp_path):
    """Callback results drop intent, optional, and value; arguments keep them."""
    from numpy.f2py import auxfuncs, crackfortran
    from numpy.f2py.f2py2e import run_main

    src = tmp_path / "gh20157.f90"
    src.write_text(textwrap.dedent("""\
        subroutine gh20157(f, y, r)
          external f
          integer(8), dimension(:) :: y
          integer(8), intent(out) :: r
          r = f(0)
          r = r + sum(y)
        end subroutine gh20157

        subroutine gh20157_opt(f, y, r)
          external f
          integer, dimension(:) :: y
          integer, optional :: r
          r = f(0)
          r = r + sum(y)
        end subroutine gh20157_opt

        subroutine gh20157_val(f, y, r)
          external f
          integer, dimension(:) :: y
          integer, value :: r
          r = f(0)
          r = r + sum(y)
        end subroutine gh20157_val

        subroutine gh20157_inout(f, r)
          external f
          integer, intent(inout) :: r
          r = f(r)
        end subroutine gh20157_inout

        subroutine gh20157_optself(f, r)
          external f
          integer, optional :: r
          r = f(r)
        end subroutine gh20157_optself

        subroutine gh20157_outself(f, r)
          external f
          integer, intent(out) :: r
          r = f(r)
        end subroutine gh20157_outself
        """))
    with util.switchdir(tmp_path):
        try:
            run_main(["--lower", "-m", "gh20157", str(src)])
        finally:
            crackfortran.reset_global_f2py_vars()
            auxfuncs.wrapfuncs = 1

    wrapper_path = tmp_path / "gh20157-f2pywrappers2.f90"
    wrapper = wrapper_path.read_text()
    callback, host = _gh20157_r_decls(wrapper)
    assert callback == [
        "integer(kind=8) :: r",
        "integer :: r",
        "integer :: r",
    ]
    assert "integer(kind=8), intent(out) :: r" in host
    assert "integer, optional :: r" in host
    assert "integer, value :: r" in host
    csrc = (tmp_path / "gh20157module.c").read_text()
    assert "r : in/output rank-0 array(int,'i')" in csrc
    assert "def f([r]): return r" in csrc
    assert "def f(): return r,r" in csrc

    gfortran = shutil.which("gfortran")
    if gfortran is None:
        pytest.skip("gfortran is not available")
    compiled = subprocess.run(
        [gfortran, "-std=f2008", "-c", str(wrapper_path)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.slow
class TestGH18335(util.F2PyTest):
    """The reproduction of the reported issue requires specific input that
    extensions may break the issue conditions, so the reproducer is
    implemented as a separate test class. Do not extend this test with
    other tests!
    """
    sources = [util.getpath("tests", "src", "callback", "gh18335.f90")]

    def test_gh18335(self):
        def foo(x):
            x[0] += 1

        r = self.module.gh18335(foo)
        assert r == 123 + 1


@pytest.mark.slow
class TestGH25211(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh25211.f"),
               util.getpath("tests", "src", "callback", "gh25211.pyf")]
    module_name = "callback2"

    def test_gh25211(self):
        def bar(x):
            return x * x

        res = self.module.foo(bar)
        assert res == 110


@pytest.mark.slow
@pytest.mark.xfail(condition=(platform.system().lower() == 'darwin'),
                   run=False,
                   reason="Callback aborts cause CI failures on macOS")
class TestCBFortranCallstatement(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh26681.f90")]
    options = ['--lower']

    def test_callstatement_fortran(self):
        with pytest.raises(ValueError, match='helpme') as exc:
            self.module.mypy_abort = self.module.utils.my_abort
            self.module.utils.do_something('helpme')
