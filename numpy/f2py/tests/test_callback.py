import math
import platform
import subprocess
import re
import sys
import textwrap
import threading
import time
import traceback
from pathlib import Path

import pytest

import numpy as np
from numpy.f2py import crackfortran
from numpy.f2py.f2py2e import main as f2pycli
from numpy.f2py.f2py2e import run_main
from numpy.testing import HAS_REFCOUNT, IS_64BIT, assert_array_equal
from numpy.testing._private.utils import requires_memory
from numpy.f2py import crackfortran, f2py2e

from . import util

LEN_BEYOND_INT = np.iinfo(np.intc).max + 1

LEN_BEYOND_INT = np.iinfo(np.intc).max + 1


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


def test_gh20157_external_dummy_interface(tmp_path, monkeypatch):
    # The interface body for gh17797 gives its EXTERNAL dummy `f` the
    # EXTERNAL attribute once, not again through an interface body.
    src = util.getpath("tests", "src", "callback", "gh17797.f90")
    monkeypatch.setattr(sys, "argv", ["f2py", str(src), "-m", "gh20157"])
    try:
        with util.switchdir(tmp_path):
            f2pycli()
        wrapper = (tmp_path / "gh20157-f2pywrappers2.f90").read_text()
        stmts = [line.strip() for line in wrapper.splitlines()]
        assert stmts.count("external f") == 2
        assert not [s for s in stmts if s.startswith("function f(")]
    finally:
        # f2pycli leaves the parser's lowercase flag unset.
        crackfortran.reset_global_f2py_vars()


@pytest.mark.slow
class TestGH20157(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh20157.f90")]

    def test_gh20157(self):
        def incr(x):
            return x + 123

        y = np.array([1, 2, 3], dtype=np.int64)
        assert self.module.gh20157(incr, y) == 123 + 1 + 2 + 3


@pytest.mark.slow
class TestF90CallbackSignature(TestF90Callback):
    # gh17797.pyf is the signature that f2py -h writes for gh17797.f90
    sources = [util.getpath("tests", "src", "callback", "gh17797.f90"),
               util.getpath("tests", "src", "callback", "gh17797.pyf")]
    module_name = "gh17797"


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
class TestGH16357(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh16357.f90")]

    @pytest.mark.parametrize("extra_args", [(), (3,), (3, 4)])
    def test_varargs(self, extra_args):
        received = []

        def fun(*args):
            received.append(args)

        self.module.gh16357(fun, 1, 2, fun_extra_args=extra_args)
        assert received == [(1, 2, *extra_args)]

    @pytest.mark.parametrize("extra_args", [(), (3,), (3, 4)])
    def test_positional_and_varargs(self, extra_args):
        received = []

        def fun(a, *args):
            received.append((a, args))

        self.module.gh16357(fun, 1, 2, fun_extra_args=extra_args)
        assert received == [(1, (2, *extra_args))]

    def test_defaults_and_varargs(self):
        received = []

        def fun(a, b=5, c=6, *args):
            received.append((a, b, c, args))

        self.module.gh16357(fun, 1, 2, fun_extra_args=(3, 4))
        assert received == [(1, 2, 3, (4,))]

    def test_method_varargs(self):
        received = []

        class A:
            def method(self, a, *args):
                received.append((a, args))

        self.module.gh16357(A().method, 1, 2, fun_extra_args=(3,))
        assert received == [(1, (2, 3))]

    def test_method_varargs_only(self):
        received = []

        class A:
            def method(*args):
                received.append(args[1:])

        self.module.gh16357(A().method, 1, 2, fun_extra_args=(3,))
        assert received == [(1, 2, 3)]

    def test_varargs_and_varkeywords(self):
        received = []

        def fun(*args, **kwargs):
            received.append((args, kwargs))

        self.module.gh16357(fun, 1, 2, fun_extra_args=(3,))
        assert received == [((1, 2, 3), {})]

    def test_varargs_too_few_arguments(self):
        error = getattr(self.module, f"_{self.module.__name__}_error")
        with pytest.raises(error):
            self.module.gh16357(lambda a, b, c, *args: None, 1, 2)


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
class TestGH18767(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh18767.f90"),
               util.getpath("tests", "src", "callback", "gh18767.pyf")]
    module_name = "gh18767_callback"

    def test_inout_string_callback(self):
        seen = []

        def cb(t):
            seen.append((t.dtype, t.shape))
            t[:] = np.frombuffer(b"HELLO", dtype="S1")

        s = np.array(b"hello", dtype="S5")
        self.module.call_cb(cb, s)
        assert seen == [(np.dtype("S1"), (5,))]
        assert s.item() == b"HELLO"

    def test_string_function_callback(self):
        f = self.module.call_string_function
        assert f(lambda: b"world") == b"world"
        assert f(self.module.hello._cpointer) == b"hello"

    @pytest.mark.skipif(not IS_64BIT, reason="needs objects larger than a C int")
    @requires_memory(free_bytes=2 * LEN_BEYOND_INT)
    def test_inout_string_callback_beyond_int(self):
        seen = []

        def cb(t):
            seen.append(t.shape)
            t[-1] = b"Z"

        s = np.full(LEN_BEYOND_INT, b"a", dtype="S1")
        s[-1] = b"z"
        self.module.call_cb(cb, s)
        assert seen == [(LEN_BEYOND_INT,)]
        assert (s[0], s[-1]) == (b"a", b"Z")
class TestGH28605(util.F2PyTest):
    # EXTERNAL dummies that the routine never calls must still be wrapped
    sources = [util.getpath("tests", "src", "callback", "gh28605.f90")]

    def test_typed(self):
        assert self.module.tcb(lambda: 0.0) == 1.0

    def test_untyped(self):
        assert self.module.scb(lambda: None) == 2

    def test_passed_on(self):
        assert self.module.forward(lambda: 2.5) == 2.5


class TestGH28605Signature:
    def test_no_arguments(self):
        fpath = util.getpath("tests", "src", "callback", "gh28605.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        cbs = {m["name"]: m["body"][0]["body"][0]
               for m in mod if "__user__" in m["name"]}
        for module, block, name in [("tcb", "function", "p"),
                                    ("scb", "subroutine", "s"),
                                    ("forward", "function", "g")]:
            cb = cbs[f"{module}__user__routines"]
            assert (cb["block"], cb["name"], cb["args"]) == (block, name, [])
        p = cbs["tcb__user__routines"]["vars"]["p"]
        assert (p["typespec"], p["kindselector"]) == ("real", {"*": "8"})
        pyf = crackfortran.crack2fortran(mod)
        assert "use tcb__user__routines" in pyf
        # forward does not call g. It passes g to callee, and callee calls
        # f with no arguments, so g reuses that signature.
        fwd = next(b for b in mod if b["name"] == "forward")
        assert fwd["use"]["callee__user__routines"]["map"] == {"g": "f"}
        assert "forward__user__routines" not in cbs
        assert "use callee__user__routines, g=>f" in pyf
        fsig = cbs["callee__user__routines"]
        assert (fsig["block"], fsig["name"], fsig["args"]) == ("function", "f", [])
        assert "use tcb__user__routines" in crackfortran.crack2fortran(mod)

    def test_signature_file_kept(self):
        fpath = util.getpath("tests", "src", "callback", "gh28605.pyf")
        mod = crackfortran.crackfortran([str(fpath)])
        assert [m["name"] for m in mod] == ["gh28605__user__routines",
                                           "gh28605"]
class TestGH2981(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh2981.f")]

    @pytest.mark.parametrize("wrap", [
        list,
        np.asarray,
        lambda y: (y,),
        lambda y: (y, "ignored"),
    ], ids=["list", "ndarray", "1-tuple", "extra-item"])
    def test_return_single_array(self, wrap):
        x = np.array([1.0, 2.0, 3.0])
        y = self.module.compute(lambda x: wrap(x + 1), x)
        np.testing.assert_array_equal(y, x + 1)

    def test_return_tuple_as_array(self):
        # The tuple is unpacked into the return values, so 2.0 is taken as y.
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=(
                r"call-back function fun returned a tuple of 3 items, which"
                r" f2py unpacks into the return values of"
                r" 'def fun\(x,\[n\]\): return y'")) as excinfo:
            self.module.compute(lambda x: tuple(x + 1), x)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "0-th dimension must be 3" in str(excinfo.value.__cause__)


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


def test_gh7577_interface_dummy_is_external():
    from numpy.f2py.crackfortran import crackfortran

    src = util.getpath("tests", "src", "callback", "gh7577.f90")

    def find(block):
        if isinstance(block, list):
            for item in block:
                found = find(item)
                if found is not None:
                    return found
            return None
        if isinstance(block, dict):
            if block.get("name") == "foo" and block.get("block") == "function":
                return block
            return find(block.get("body", []))
        return None

    foo = find(crackfortran([str(src)]))
    assert foo is not None
    assert "func_complex" in foo.get("externals", [])


@pytest.mark.slow
def test_gh7577_complex_callback_return(tmp_path):
    if not util.has_fortran_compiler():
        pytest.skip("No Fortran compiler available")
    src = util.getpath("tests", "src", "callback", "gh7577.f90")
    (tmp_path / "gh7577.f90").write_text(src.read_text())
    code = """
import sys
sys.path.insert(0, ".")
sys.argv = ["f2py", "-c", "-m", "gh7577", "gh7577.f90",
            "--backend", "meson", "--freethreading-compatible"]
from numpy.f2py.f2py2e import main
main()
import gh7577
def sq(z):
    return z * z
got = gh7577.gh7577.foo(1j, sq)
if abs(got - (-1 + 0j)) > 1e-6:
    raise SystemExit(f"got={got!r}")
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
class TestGH22451(util.F2PyTest):
    # gh-22451: an EXTERNAL dummy that is only passed on to another routine
    sources = [util.getpath("tests", "src", "callback", "gh22451.f90")]

    def test_identity(self):
        # fun1 calls the callback with the integer literal 1.
        argument = 1
        assert self.module.fun2(argument, lambda x: x) == float(argument)

    def test_scaled(self):
        # Distinct from 1 so a dropped callback return cannot pass.
        argument = 1
        scale = 10.0
        result = self.module.fun2(argument, lambda x: scale * x)
        assert result == scale * argument


class TestGH22451Signature:
    def test_reuses_callee_signature(self):
        fpath = util.getpath("tests", "src", "callback", "gh22451.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        fun2 = next(b for b in mod if b["name"] == "fun2")
        assert fun2["use"]["fun1__user__routines"]["map"] == {"myfun2": "myfun1"}
        names = [m["name"] for m in mod]
        assert "fun1__user__routines" in names
        assert "fun2__user__routines" not in names
        pyf = crackfortran.crack2fortran(mod)
        assert "use fun1__user__routines, myfun2=>myfun1" in pyf

    def test_interface_block_wins(self, tmp_path):
        src = tmp_path / "iface.f90"
        src.write_text(textwrap.dedent("""\
            function fun2(myfun2)
                implicit none
                real(8) :: fun2
                interface
                    function myfun2(x)
                        real(8) :: myfun2
                        real(8) :: x
                    end function myfun2
                end interface
                fun2 = fun1(myfun2)
            end function fun2

            function fun1(myfun1)
                implicit none
                real(8) :: fun1
                real(8), external :: myfun1
                fun1 = myfun1(1)
            end function fun1
            """))
        mod = crackfortran.crackfortran([str(src)])
        pyf = crackfortran.crack2fortran(mod)
        assert "function myfun2(x)" in pyf
        assert "myfun2=>myfun1" not in pyf

    def test_missing_callee_keeps_no_arguments(self, tmp_path):
        src = tmp_path / "missing.f90"
        src.write_text(textwrap.dedent("""\
            function t(p)
                implicit none
                real(8) :: t
                real(8), external :: p
                call missing(p)
                t = 1.0
            end function t
            """))
        mod = crackfortran.crackfortran([str(src)])
        cbs = {m["name"]: m["body"][0]["body"][0]
               for m in mod if "__user__" in m["name"]}
        cb = cbs["t__user__routines"]
        assert (cb["block"], cb["name"], cb["args"]) == ("function", "p", [])


@pytest.mark.slow
class TestGH24284(util.F2PyTest):
    # gh-24284: fixed-form EXTERNAL dummies passed down, then called
    sources = [util.getpath("tests", "src", "callback", "gh24284.f")]

    def test_both_called(self):
        # divam calls the typed callback with this integer.
        n_value = 4
        # Value the untyped callback must then receive.
        typed_return = 3.0
        seen = {}

        def typed_cb(n):
            seen["typed"] = n
            return typed_return

        def untyped_cb(n, y):
            seen["untyped_n"] = n
            seen["untyped_y"] = y

        self.module.diva(typed_cb, untyped_cb, n_value)
        assert int(seen["typed"]) == n_value
        assert int(seen["untyped_n"]) == n_value
        assert seen["untyped_y"] == typed_return
@pytest.mark.slow
class TestForwardedCallbackArguments(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh22451.f90")]

    @pytest.mark.parametrize("name", ["fun1", "fun2", "fun3", "named_forward", "nested_forward"])
    def test_integer_argument(self, name):
        received = []

        def callback(value):
            received.append(value)
            return value

        expected = 3 if name == "named_forward" else 1
        function = getattr(self.module, name)
        result = function(0.0, callback) if name in ("fun2", "fun3") else function(callback)
        assert received == [expected]
        assert result == expected

    def test_subroutine_forwarding(self):
        received = []

        def callback(value):
            received.append(value)
            return value * 3

        assert self.module.forward_sub(callback) == 6
        assert received == [2]


class TestForwardedCallbackSignatures:
    @pytest.mark.parametrize("selected", [False, True])
    def test_generated_callback_argument_counts(self, tmp_path, monkeypatch, selected):
        source = util.getpath("tests", "src", "callback", "gh22451.f90")
        monkeypatch.chdir(tmp_path)
        module = "gh22451_generated"
        arguments = ["-m", module, str(source)]
        if selected:
            arguments += ["only:", "fun2", ":"]
        f2py2e.run_main(arguments)
        generated = (tmp_path / f"{module}module.c").read_text()
        callback_arguments = re.findall(
            r"create_cb_arglist\([^,\n]+,[^,\n]+,\s*(\d+),\s*(\d+),",
            generated,
        )
        expected_callbacks = 1 if selected else 8
        assert callback_arguments == [("1", "0")] * expected_callbacks

    def test_chained_argument_type(self):
        source = util.getpath("tests", "src", "callback", "gh22451.f90")
        blocks = crackfortran.crackfortran([str(source)])
        callbacks = {module["name"]: module["body"][0]["body"][0]
                     for module in blocks if "__user__" in module["name"]}
        for routine, callback in [("fun1", "myfun1"), ("fun2", "myfun2"), ("fun3", "myfun3")]:
            declaration = callbacks[f"{routine}__user__routines"]
            assert declaration["name"] == callback
            assert len(declaration["args"]) == 1
            argument = declaration["args"][0]
            assert declaration["vars"][argument]["typespec"] == "integer"

    def test_incompatible_consumers(self, tmp_path):
        source = tmp_path / "incompatible.f90"
        source.write_text(textwrap.dedent("""\
            real(8) function integer_consumer(cb)
                real(8), external :: cb
                integer_consumer = cb(1)
            end function
            real(8) function real_consumer(cb)
                real(8), external :: cb
                real_consumer = cb(1d0)
            end function
            real(8) function forward(cb)
                real(8), external :: cb,integer_consumer,real_consumer
                forward = integer_consumer(cb) + real_consumer(cb)
            end function
            """))
        from numpy.f2py.auxfuncs import F2PYError

        with pytest.raises(F2PYError, match="Incompatible interfaces for forwarded callback"):
            crackfortran.crackfortran([str(source)])

    def test_call_in_string_literal(self, tmp_path):
        source = tmp_path / "literal.f90"
        source.write_text(textwrap.dedent("""\
            real(8) function consumer(cb)
                real(8), external :: cb
                consumer = cb(1)
            end function
            real(8) function unused(cb)
                real(8), external :: cb
                character(20) :: message
                message = 'consumer(cb)'
                unused = 0d0
            end function
            """))
        blocks = crackfortran.crackfortran([str(source)])
        module = next(block for block in blocks if block["name"] == "unused__user__routines")
        assert module["body"][0]["body"][0]["args"] == []


@pytest.mark.slow
class TestSelectedForwardedCallbackArguments(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh22451.f90")]
    options = ["only:", "fun2", ":"]

    def test_selected_wrapper(self):
        received = []

        def callback(value):
            received.append(value)
            return value

        assert self.module.fun2(0.0, callback) == 1
        assert received == [1]
        assert not hasattr(self.module, "fun1")
@pytest.mark.slow
class TestForwardedCallbackGH31248(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh31248_arg.f90"),
               util.getpath("tests", "src", "callback", "gh31248_arg.pyf")]
    module_name = "gh31248_arg"

    def test_arguments(self):
        seen = []
        self.module.solve(3, lambda step, x: seen.append((step, x)))
        assert seen == [(1, 0.5), (2, 0.25), (3, 0.125)]

    def test_exception_abandons_fortran_call(self):
        seen = []

        def monitor(step, x):
            seen.append(step)
            raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            self.module.solve(3, monitor)
        assert seen == [1]


@pytest.mark.slow
class TestNamedCallbackGH31248(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh31248_named.f90"),
               util.getpath("tests", "src", "callback", "gh31248_named.pyf")]
    module_name = "gh31248_named"

    def test_undefined(self):
        with pytest.raises(Exception, match="Callback on_step not defined"):
            self.module.run(3)

    def test_arguments(self):
        seen = []
        self.module.on_step = lambda step, x: seen.append((step, x))
        try:
            self.module.run(3)
            self.module.report(7, 1.5)
        finally:
            del self.module.on_step
        assert seen == [(1, 0.5), (2, 0.25), (3, 0.125), (7, 1.5)]

    def test_exception_abandons_fortran_call(self):
        seen = []

        def on_step(step, x):
            seen.append(step)
            raise RuntimeError("stop")

        self.module.on_step = on_step
        try:
            with pytest.raises(RuntimeError, match="stop"):
                self.module.run(3)
        finally:
            del self.module.on_step
        assert seen == [1]
