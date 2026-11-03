import math
import platform
import sys
import textwrap
import threading
import time
import traceback

import pytest

import numpy as np
from numpy.f2py import crackfortran

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
                                    ("scb", "subroutine", "s")]:
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

    def test_signature_file_kept(self):
        fpath = util.getpath("tests", "src", "callback", "gh28605.pyf")
        mod = crackfortran.crackfortran([str(fpath)])
        assert [m["name"] for m in mod] == ["gh28605__user__routines",
                                           "gh28605"]


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
