import math
import platform
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


@pytest.mark.slow
class TestCallbackIntrinsicArg(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine call_size(f, b, c)
  external f
  double precision, intent(in) :: b(:)
  double precision, intent(out) :: c
  call f(b, size(b), c)
end subroutine

subroutine call_ubound(f, b, c)
  external f
  double precision, intent(in) :: b(:)
  double precision, intent(out) :: c
  call f(b, ubound(b, 1), c)
end subroutine

integer function call_size_function(f, b, n)
  integer f
  external f
  integer, intent(in) :: n
  double precision, intent(in) :: b(n)
  call_size_function = f(size(b))
end function

subroutine call_int_kind8(f, x, c)
  external f
  double precision, intent(in) :: x
  integer(8), intent(out) :: c
  call f(int(x, kind=8), c)
end subroutine

subroutine call_mod_real(f, x, c)
  external f
  real, intent(in) :: x
  real, intent(out) :: c
  call f(mod(x, 2.0), c)
end subroutine

subroutine call_mod_real8(f, x, c)
  external f
  real(8), intent(in) :: x
  real(8), intent(out) :: c
  call f(mod(x, 2.0_8), c)
end subroutine

subroutine call_max_real(f, a, b, c)
  external f
  real, intent(in) :: a, b
  real, intent(out) :: c
  call f(max(a, b), c)
end subroutine

subroutine call_size_array(f, size, n, c)
  external f
  integer, intent(in) :: n
  double precision, intent(in) :: size(n)
  double precision, intent(out) :: c
  call f(size(2), c)
end subroutine

subroutine call_size_times_real(f, b, c)
  external f
  double precision, intent(in) :: b(:)
  double precision, intent(out) :: c
  call f(size(b) * (2.5), c)
end subroutine

subroutine call_array_and_element(f, a, c)
  external f
  double precision, intent(in) :: a(3)
  double precision, intent(out) :: c
  call f(a, a(2), c)
end subroutine
"""

    @pytest.mark.parametrize("name", ["call_size", "call_ubound"])
    def test_integer_intrinsic_argument(self, name):
        assert getattr(self.module, name)(lambda b, n: n, [1.0, 2.0, 3.0]) == 3

    def test_integer_intrinsic_function_argument(self):
        assert self.module.call_size_function(lambda n: n, [1.0, 2.0, 3.0]) == 3

    def test_integer_intrinsic_kind8(self):
        assert self.module.call_int_kind8(lambda n: n, 5.0e9) == 5_000_000_000

    def test_arg_typed_intrinsics(self):
        assert self.module.call_mod_real(lambda v: v, 5.0) == 1.0
        assert self.module.call_mod_real8(lambda v: v, 5.5) == 1.5
        assert self.module.call_max_real(lambda v: v, 2.5, 5.5) == 5.5

    def test_user_array_named_size(self):
        assert self.module.call_size_array(lambda v: v, [1.5, 2.5, 3.5]) == 2.5

    def test_compound_expression_argument(self):
        assert self.module.call_size_times_real(lambda v: v, [1.0, 2.0, 3.0]) == 7.5

    def test_array_argument_keeps_dimension(self):
        seen = []

        def f(a, a2):
            seen.append((list(a), a2))
            return a2

        assert self.module.call_array_and_element(f, [1.0, 2.0, 3.0]) == 2.0
        assert seen == [([1.0, 2.0, 3.0], 2.0)]
