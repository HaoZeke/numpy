import os
import subprocess
import sys
import textwrap

import pytest

from numpy.testing import HAS_SUBPROCESSES

from . import util


@pytest.mark.skipif(
    not HAS_SUBPROCESSES, reason="platform cannot start subprocesses"
)
def test_import_does_not_run_meson():
    # gh-25239: importing the test utilities must not start meson
    code = textwrap.dedent(f"""
        import subprocess
        import sys
        sys.path = {sys.path!r}

        def no_subprocess(*args, **kwargs):
            raise AssertionError(f"subprocess started on import: {{args!r}}")

        subprocess.run = no_subprocess
        subprocess.Popen = no_subprocess
        import numpy.f2py.tests.util
    """)
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(
    not HAS_SUBPROCESSES, reason="platform cannot start subprocesses"
)
def test_missing_meson_does_not_skip_module(tmp_path):
    code = textwrap.dedent(f"""
        import os
        import sys
        os.environ["PATH"] = {str(tmp_path)!r}
        sys.path = {sys.path!r}
        import pytest
        try:
            import numpy.f2py.tests.util
        except pytest.skip.Exception as exc:
            raise SystemExit(
                f"module skip {{exc.allow_module_level}} {{exc}}"
            )
        print("IMPORTED")
    """)
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    assert "IMPORTED" in res.stdout


@pytest.mark.skipif(
    not HAS_SUBPROCESSES, reason="platform cannot start subprocesses"
)
def test_session_fixture_skips_build_not_pure_test(tmp_path):
    # Fixed-form source, accepted by gfortran -fsyntax-only.
    fortran = "      subroutine p(x)\n      integer x\n      x = 1\n      end\n"
    testfile = tmp_path / "test_probe_scope.py"
    testfile.write_text(textwrap.dedent(f"""
        pytest_plugins = ["numpy.f2py.tests.conftest"]
        import numpy.f2py.tests.util as util

        def test_pure():
            assert util.checker.compilers_checked is False

        class TestBuild(util.F2PyTest):
            code = {fortran!r}
            def test_build(self):
                assert self.module is not None
    """))
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)
    res = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(testfile),
            "-rs", "-q", "--tb=short", "-p", "no:cacheprovider",
            "--override-ini", "addopts=",
        ],
        capture_output=True, text=True, env=env,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "1 passed" in res.stdout
    assert "1 skipped" in res.stdout
    assert "meson not present" in out
