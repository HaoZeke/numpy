import subprocess
import sys
import textwrap
import types

import pytest

from numpy.testing import HAS_SUBPROCESSES

from . import util


@pytest.mark.skipif(not HAS_SUBPROCESSES, reason="platform cannot start subprocesses")
def test_import_does_not_run_meson():
    # gh-25239, gh-25447: probing for compilers on import slowed down test
    # collection and turned probe failures into collection errors
    code = textwrap.dedent(f"""
        import subprocess
        import sys
        sys.path = {sys.path!r}

        def no_subprocess(*args, **kwargs):
            raise AssertionError(f"subprocess started on import: {{args}}")

        subprocess.Popen = no_subprocess
        import numpy.f2py.tests.util
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(
    sys.platform == "win32", reason="No Fortran tests on Windows (Issue #25134)"
)
def test_missing_meson_skips_test_not_module(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="meson not present") as exc:
        util.CompilerChecker().check_compilers()
    assert not exc.value.allow_module_level


def test_build_module_rejects_imported_name(monkeypatch):
    # Building under an imported name used to return the earlier module
    name = "_test_util_imported_name"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    src = util.getpath("tests", "src", "regression", "inout.f90")
    with pytest.raises(RuntimeError, match="already in use"):
        util.build_module([src], module_name=name)


class TestNoModuleNoCompilerCheck(util.F2PyTest):
    # Overrides the session fixture: classes that build nothing must not
    # depend on meson or compilers
    @pytest.fixture(scope="class")
    @classmethod
    def check_compilers(cls):
        pytest.fail("check_compilers requested without sources or code")

    def test_no_module(self):
        assert self.module is None
