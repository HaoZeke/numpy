import math
import platform
import re
import subprocess
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


def _generate_module_c(tmp_path, source, mname):
    """Run f2py via subprocess to emit module.c only (no compile)."""
    fpath = tmp_path / f"{mname}.f90"
    fpath.write_text(textwrap.dedent(source), encoding="ascii")
    cmd = [sys.executable, "-m", "numpy.f2py", "-m", mname, str(fpath)]
    subprocess.check_call(cmd, cwd=tmp_path)
    cpath = tmp_path / f"{mname}module.c"
    assert cpath.is_file(), f"expected generated C wrapper at {cpath}"
    return cpath.read_text(encoding="utf-8")


class TestExternalCallbackTypedefCodegen:
    """Compiler-free codegen checks for external-without-interface callbacks.

    Root cause (gh-22451 / gh-24284 / gh-28605): an F77-style ``external``
    dummy never called in the parent body gets no ``lcb_map`` entry, so the
    wrapper emits an undeclared ``*_t`` typedef and leaves ``#maxnofargs#`` /
    ``#nofoptargs#`` unsubstituted.
    """

    def test_gh22451_forwarded_external_emits_callback_typedef(self, tmp_path):
        # 15-line MWE from gh-22451: myfun2 is passed onward, never called.
        source = """
        function fun1(myfun1)
            implicit none
            real(8) :: fun1
            real(8), external :: myfun1
            fun1 = myfun1(1)
            return
        end function fun1

        function fun2(b, myfun2)
            implicit none
            real(8) :: b, fun2
            real(8), external :: myfun2, fun1
            b = fun1(myfun2)
            fun2 = b
            return
        end function fun2
        """
        csrc = _generate_module_c(tmp_path, source, "gh22451")
        assert "#maxnofargs#" not in csrc
        assert "#nofoptargs#" not in csrc
        # Must not emit a bare myfun2_t; the proper cb_* name is required.
        assert re.search(r"(?<![\w])myfun2_t\b", csrc) is None
        assert "cb_myfun2_in_fun2__user__routines_t" in csrc
        assert "typedef struct" in csrc
        assert re.search(
            r"create_cb_arglist\([^;]*myfun2[^;]*,\s*0\s*,\s*0\s*,",
            csrc,
        )

    def test_gh28605_dup_external_emits_callback_typedef(self, tmp_path):
        # external p + real*8 p used to yield externals=['p','p'] and bare p_t.
        source = """
        real*8 function tcb(p)
          external p
          real*8 p, a(6), c(6)
          integer i
          a = [(i, i = 1, 6)]
          c = [(i, i = 1, 6)]
          tcb = 1.0
          return
        end function tcb
        """
        csrc = _generate_module_c(tmp_path, source, "gh28605")
        assert "#maxnofargs#" not in csrc
        assert "#nofoptargs#" not in csrc
        assert re.search(r"(?<![\w_])p_t\b", csrc) is None
        assert "cb_p_in_tcb__user__routines_t" in csrc
        assert re.search(
            r"create_cb_arglist\([^;]*p_cb[^;]*,\s*0\s*,\s*0\s*,",
            csrc,
        )

    def test_gh28605_externals_dedup_in_crack(self, tmp_path):
        fpath = tmp_path / "tcb.f90"
        fpath.write_text(textwrap.dedent("""
        real*8 function tcb(p)
          external p
          real*8 p
          tcb = 1.0
        end function tcb
        """), encoding="ascii")
        post = crackfortran.crackfortran([str(fpath)])
        tcb = next(b for b in post if b.get("name") == "tcb")
        assert tcb["externals"] == ["p"]
