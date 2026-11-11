import math
import platform
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from pathlib import Path

import pytest

import numpy as np
from numpy.f2py.f2py2e import run_main
from numpy.testing import HAS_REFCOUNT, assert_array_equal

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

    def test_hidden_callback_extra_args(self):
        self.module.global_f = lambda x, y=0: x + y
        self.module.global_f_extra_args = [10]
        try:
            assert self.module.hidden_callback(2) == 12
            # Only the extra arguments reach a callback that is not an
            # argument of the wrapper (gh18341)
            assert self.module.hidden_callback2(2) == 10
        finally:
            del self.module.global_f
            del self.module.global_f_extra_args

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

    def test_hidden_callback_undeclared(self, monkeypatch):
        # gh-18385: hidden_callback2 calls global_f without declaring it
        monkeypatch.setattr(self.module, "global_f", lambda x: x + 1,
                            raising=False)
        assert self.module.hidden_callback2(2) == 3

        monkeypatch.setattr(self.module, "global_f", lambda x, y: x * y)
        monkeypatch.setattr(self.module, "global_f_extra_args", (10,),
                            raising=False)
        assert self.module.hidden_callback(2) == 20
        assert self.module.hidden_callback2(2) == 20

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_hidden_callback_refcount(self):
        # The call-back and its extra arguments are module attributes; each
        # call must release what it took, with or without a declaration
        def global_f(*args):
            return 0

        extra_args = (object(),)
        self.module.global_f = global_f
        self.module.global_f_extra_args = extra_args
        try:
            for routine in (self.module.hidden_callback,
                            self.module.hidden_callback2):
                before = sys.getrefcount(global_f), sys.getrefcount(extra_args)
                assert routine(2) == 0
                after = sys.getrefcount(global_f), sys.getrefcount(extra_args)
                assert after == before
        finally:
            del self.module.global_f
            del self.module.global_f_extra_args


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
class TestGH18385(util.F2PyTest):
    # primary and tertiary declare the hidden call-back, secondary does not
    sources = [util.getpath("tests", "src", "callback", "gh18385.f90")]

    def test_gh18385(self, monkeypatch):
        calls = []
        monkeypatch.setattr(self.module, "callback",
                            lambda a, b: calls.append((a, b)), raising=False)
        self.module.primary()
        self.module.secondary()
        self.module.tertiary()
        assert calls == [(1, 2), (3, 4), (5, 6)]

    def test_nested(self, monkeypatch):
        calls = []

        def callback(a, b):
            calls.append((a, b))
            if a == 1:
                self.module.tertiary()
            elif a == 5:
                self.module.secondary()

        monkeypatch.setattr(self.module, "callback", callback, raising=False)
        self.module.primary()
        assert calls == [(1, 2), (5, 6), (3, 4)]

    def test_raises(self, monkeypatch):
        def callback(a, b):
            raise ValueError(f"from {a}")

        monkeypatch.setattr(self.module, "callback", callback, raising=False)
        for routine, a in [(self.module.primary, 1),
                           (self.module.secondary, 3),
                           (self.module.tertiary, 5)]:
            with pytest.raises(ValueError, match=f"from {a}"):
                routine()


@pytest.mark.slow
class TestGH18385PythonTLS(TestGH18385):
    options = ["-DF2PY_USE_PYTHON_TLS"]


def _signature_file(test, source):
    # build_module copies the sources into the module directory, so the
    # signature file is written next to it rather than into it
    pyf = Path(util.get_module_dir()) / "signatures" / f"{test.module_name}.pyf"
    if not pyf.exists():
        pyf.parent.mkdir(exist_ok=True)
        run_main(["-h", str(pyf), "-m", test.module_name, str(source)])
    return [pyf, source]


@pytest.mark.slow
class TestF77CallbackSignatureFile(TestF77Callback):
    # foo.f through the signature file that f2py -h writes for it
    @property
    def sources(self):
        return _signature_file(
            self, util.getpath("tests", "src", "callback", "foo.f"))


@pytest.mark.slow
class TestGH18385SignatureFile(TestGH18385):
    @property
    def sources(self):
        return _signature_file(
            self, util.getpath("tests", "src", "callback", "gh18385.f90"))


@pytest.mark.slow
class TestGH18385ArrayCallback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback",
                           "matching_callback_shapes.f90")]

    def test_matching_array_shapes(self, monkeypatch):
        observed = []

        def callback(values):
            observed.append(values.copy())

        monkeypatch.setattr(self.module, "callback", callback, raising=False)
        self.module.primary()
        self.module.secondary()

        assert len(observed) == 2
        expected = np.arange(1, 7, dtype=np.float64).reshape((2, 3), order="F")
        assert_array_equal(observed[0], expected)
        assert_array_equal(observed[1], expected + 10)


@pytest.mark.parametrize("second_dimensions", ["3, 2", "6"])
def test_gh18385_array_signature_mismatch(tmp_path, second_dimensions):
    fpath = tmp_path / "gh18385_array_mismatch.f90"
    fpath.write_text(textwrap.dedent(f"""\
        subroutine primary()
          implicit none
          double precision values(2, 3)
          external callback
          !f2py intent(callback, hide) :: callback
          !f2py intent(in) :: values
          call callback(values)
        end subroutine primary

        subroutine secondary()
          implicit none
          double precision values({second_dimensions})
          external callback
          !f2py intent(callback, hide) :: callback
          !f2py intent(in) :: values
          call callback(values)
        end subroutine secondary
        """))
    code = f"import sys; sys.path = {sys.path!r}; import numpy.f2py; numpy.f2py.main()"
    res = subprocess.run([sys.executable, "-c", code, "-m", "mismatch", str(fpath)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode != 0
    assert "F2PYError" in res.stderr
    assert "intent(callback) external 'callback'" in res.stderr
    assert "different signatures" in res.stderr


def test_gh18385_signature_mismatch(tmp_path):
    # One trampoline serves the two routines, so they must agree on it
    fpath = tmp_path / "gh18385_mismatch.f90"
    fpath.write_text(textwrap.dedent("""\
        subroutine one()
          external cb
          !f2py intent(callback, hide) :: cb
          call cb(1)
        end subroutine one

        subroutine two()
          external cb
          !f2py intent(callback, hide) :: cb
          call cb(1.0)
        end subroutine two
        """))
    code = f"import sys; sys.path = {sys.path!r}; import numpy.f2py; numpy.f2py.main()"
    res = subprocess.run([sys.executable, "-c", code, "-m", "mismatch", str(fpath)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode != 0
    assert "F2PYError" in res.stderr
    assert "different signatures" in res.stderr


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
class TestStringOutCallback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "string_out.f"),
               util.getpath("tests", "src", "callback", "string_out.pyf")]
    module_name = "callback_string_out"

    def test_string_out(self):
        assert self.module.string_out(lambda: b"hello") == b"hello"
        assert self.module.string_out(lambda: b"hi") == b"hi"

    def test_string_out_not_bytes(self, capfd):
        err = getattr(self.module, f"_{self.module_name}_error")
        with pytest.raises(err, match="string object expected"):
            self.module.string_out(lambda: "hello")
        assert (f"{self.module_name}.error is related to hello\n"
                in capfd.readouterr().err)


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
