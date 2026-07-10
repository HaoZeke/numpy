import math
import platform
import re
import subprocess
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

    def test_raising_callback_quiet_stderr(self, capfd):
        # gh-5098: a raising callback must propagate the exception
        # without dumping diagnostics to stderr
        def callback():
            raise RuntimeError("failing on purpose")

        with pytest.raises(RuntimeError, match="failing on purpose"):
            self.module.t(callback)
        assert capfd.readouterr().err == ""

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


class TestF90CallbackCodegen:
    """Verify that generated Fortran 90 callback wrappers do not contain
    conflicting ``external`` declarations when an interface block for the
    same name already exists (gh-20157).

    Strict Fortran compilers (e.g. Intel ifort/ifx >= 2019) reject code
    where a dummy procedure is both declared ``external`` and given an
    explicit interface in the same scope.
    """

    def test_gh20157_no_conflicting_external(self):
        """The wrapper for a function with a callback and assumed-shape
        array should use an interface block for the callback instead of
        a bare ``external`` declaration."""
        import os
        import subprocess
        import tempfile

        src = textwrap.dedent("""\
            function gh17797(f, y) result(r)
              external f
              integer(8) :: r, f
              integer(8), dimension(:) :: y
              r = f(0)
              r = r + sum(y)
            end function gh17797
        """)
        with tempfile.NamedTemporaryFile(
            suffix='.f90', mode='w', delete=False
        ) as tf:
            tf.write(src)
            tf.flush()
            tmpdir = tempfile.mkdtemp()
            # Run f2py in a subprocess to avoid polluting crackfortran
            # global state in the test process.
            code = (
                "import sys; sys.path = %r; "
                "import numpy.f2py; numpy.f2py.main()"
            ) % sys.path
            cmd = [
                sys.executable, "-c", code,
                tf.name, '--build-dir', tmpdir, '-m', '_test_gh20157'
            ]
            p = subprocess.run(
                cmd, capture_output=True, text=True
            )
            assert p.returncode == 0, (
                f"f2py failed:\n{p.stdout}\n{p.stderr}"
            )

        wrapper_file = os.path.join(
            tmpdir, '_test_gh20157-f2pywrappers2.f90')
        assert os.path.exists(wrapper_file), (
            "F90 wrapper file was not generated")

        with open(wrapper_file) as fh:
            wrapper = fh.read()

        # Split the wrapper into lines for analysis
        lines = wrapper.split('\n')

        # Find the outer wrapper subroutine scope and check that there
        # is no bare ``external f`` at the top level.  The callback's
        # interface should be provided via an interface block instead.
        in_outer_sub = False
        interface_depth = 0
        found_external_f = False
        found_interface_f = False

        for line in lines:
            stripped = line.strip().lower()
            if 'subroutine f2pywrap' in stripped:
                in_outer_sub = True
                continue
            if not in_outer_sub:
                continue
            # Track interface block nesting
            if stripped.startswith('interface'):
                interface_depth += 1
            if stripped.startswith('end interface'):
                interface_depth -= 1
                continue
            # Only inspect the outermost scope
            if interface_depth == 0:
                if stripped == 'external f':
                    found_external_f = True
            # Check for interface block containing function f
            if interface_depth == 1:
                if 'function f(' in stripped:
                    found_interface_f = True

        assert not found_external_f, (
            "Wrapper must not use bare 'external f' when an "
            "interface block for f exists (gh-20157). "
            "Generated wrapper:\n" + wrapper
        )
        assert found_interface_f, (
            "Wrapper should contain an interface block defining "
            "callback f. Generated wrapper:\n" + wrapper
        )

    def test_f77_callback_external_preserved(self):
        """F77 callbacks without assumed-shape arrays should keep their
        ``external`` declarations and not generate interface blocks.

        This ensures the F90 callback fix (gh-20157) does not break
        the legacy F77 code path.
        """
        import os
        import subprocess
        import tempfile

        # Pure F77-style callback: no assumed-shape arrays, so no
        # need_interface, and no saved_interface with interface blocks.
        # F77 fixed format requires 6-space indentation.
        src = (
            "      integer function foo(cb, n)\n"
            "      external cb\n"
            "      integer cb, n\n"
            "      foo = cb(n)\n"
            "      return\n"
            "      end\n"
        )
        with tempfile.NamedTemporaryFile(
            suffix='.f', mode='w', delete=False
        ) as tf:
            tf.write(src)
            tf.flush()
            tmpdir = tempfile.mkdtemp()
            code = (
                "import sys; sys.path = %r; "
                "import numpy.f2py; numpy.f2py.main()"
            ) % sys.path
            cmd = [
                sys.executable, "-c", code,
                tf.name, '--build-dir', tmpdir, '-m', '_test_f77cb'
            ]
            p = subprocess.run(
                cmd, capture_output=True, text=True
            )
            assert p.returncode == 0, (
                f"f2py failed:\n{p.stdout}\n{p.stderr}"
            )

        # F77 code should not produce an F90 wrapper file
        wrapper_f90 = os.path.join(
            tmpdir, '_test_f77cb-f2pywrappers2.f90')
        wrapper_f77 = os.path.join(
            tmpdir, '_test_f77cb-f2pywrappers.f')

        # The C module source should be generated
        c_file = os.path.join(tmpdir, '_test_f77cbmodule.c')
        assert os.path.exists(c_file), (
            "C module file was not generated")

        # If an F77 wrapper exists, it should use external (not interface)
        if os.path.exists(wrapper_f77):
            with open(wrapper_f77) as fh:
                content = fh.read().lower()
            assert 'interface' not in content, (
                "F77 wrapper should not contain interface blocks. "
                f"Content:\n{content}")

    def test_f90_callback_with_explicit_shape_preserves_external(self):
        """F90 callbacks with explicit-shape (not assumed-shape) arrays
        should preserve ``external`` declarations since no interface
        block is needed for those."""
        import os
        import subprocess
        import tempfile

        src = textwrap.dedent("""\
            function bar(cb, y, n) result(r)
              external cb
              integer(8) :: r, cb
              integer :: n
              integer(8) :: y(n)
              r = cb(0) + sum(y)
            end function bar
        """)
        with tempfile.NamedTemporaryFile(
            suffix='.f90', mode='w', delete=False
        ) as tf:
            tf.write(src)
            tf.flush()
            tmpdir = tempfile.mkdtemp()
            code = (
                "import sys; sys.path = %r; "
                "import numpy.f2py; numpy.f2py.main()"
            ) % sys.path
            cmd = [
                sys.executable, "-c", code,
                tf.name, '--build-dir', tmpdir, '-m', '_test_f90_explicit'
            ]
            p = subprocess.run(
                cmd, capture_output=True, text=True
            )
            assert p.returncode == 0, (
                f"f2py failed:\n{p.stdout}\n{p.stderr}"
            )

        # With explicit-shape arrays, no F90 wrapper should be needed
        # (no need_interface), or if generated it should keep external
        wrapper_file = os.path.join(
            tmpdir, '_test_f90_explicit-f2pywrappers2.f90')
        if os.path.exists(wrapper_file):
            with open(wrapper_file) as fh:
                wrapper = fh.read()
            # If there is a wrapper, the external declaration for cb
            # should be kept since there is no assumed-shape array
            # triggering the interface replacement logic
            lines = wrapper.lower().split('\n')
            in_outer = False
            interface_depth = 0
            for line in lines:
                s = line.strip()
                if 'subroutine f2pywrap' in s:
                    in_outer = True
                    continue
                if not in_outer:
                    continue
                if s.startswith('interface'):
                    interface_depth += 1
                if s.startswith('end interface'):
                    interface_depth -= 1


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


def test_gh7577_complex_callback_codegen(tmp_path, monkeypatch):
    """Compiler-free check that complex-returning callbacks enable struct ABI."""
    fpath = util.getpath("tests", "src", "callback", "gh7577.f")
    mname = "gh7577codegen"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", f"f2py -m {mname} {fpath}".split())
    f2pycli()
    csrc = Path(f"{mname}module.c").read_text()
    define_at = csrc.index("#define F2PY_CB_RETURNCOMPLEX")
    callback_at = csrc.index("cb_callback_in_complex_cb_test__user__routines")
    assert define_at < callback_at
    assert re.search(
        r"#ifdef F2PY_CB_RETURNCOMPLEX\ncomplex_float\n#else\nvoid\n#endif\n"
        r" F_FUNC\(callback,CALLBACK\) \(\n"
        r"#ifndef F2PY_CB_RETURNCOMPLEX\n"
        r"complex_float \*return_value\n#endif",
        csrc,
    )
    assert "#ifdef F2PY_CB_RETURNCOMPLEX\n    return return_value;" in csrc


@pytest.mark.slow
class TestGH7577ComplexCallback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh7577.f")]

    def test_complex_callback_return(self):
        def square(z):
            return z * z

        r = self.module.complex_cb_test(square, 1j)
        assert r == -1 + 0j
