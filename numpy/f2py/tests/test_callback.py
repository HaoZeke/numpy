import math
import platform
import re
import sys
import textwrap
import threading
import time
import traceback

import pytest

import numpy as np
from numpy.f2py import crackfortran, f2py2e

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
                                    ("scb", "subroutine", "s"),
                                    ("forward", "function", "g")]:
            cb = cbs[f"{module}__user__routines"]
            assert (cb["block"], cb["name"], cb["args"]) == (block, name, [])
        p = cbs["tcb__user__routines"]["vars"]["p"]
        assert (p["typespec"], p["kindselector"]) == ("real", {"*": "8"})
        assert "use tcb__user__routines" in crackfortran.crack2fortran(mod)

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
