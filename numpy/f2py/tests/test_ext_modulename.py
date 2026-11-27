import importlib
import importlib.util
import platform
import sys
import sysconfig
from pathlib import Path

import pytest

from numpy.f2py.f2py2e import main as f2pycli

from . import util

SRC = """
subroutine hello(n)
  integer, intent(out) :: n
  n = 7
end subroutine hello
"""


def _build(tmp_path, monkeypatch, modulename):
    if not util.has_fortran_compiler():
        pytest.skip("CLI command needs a Fortran compiler")
    src = tmp_path / "hello.f90"
    src.write_text(SRC)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["f2py", "-c", "--backend", "meson", "-m", modulename, str(src)],
    )
    f2pycli()


@pytest.mark.slow
@pytest.mark.skipif(platform.system() != "Linux", reason="Compiler required")
def test_dashed_extension_name(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, "my-mod")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    path = tmp_path / f"my-mod{suffix}"
    spec = importlib.util.spec_from_file_location("my-mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.hello() == 7
    assert mod.__name__ == "my-mod"


@pytest.mark.slow
@pytest.mark.skipif(platform.system() != "Linux", reason="Compiler required")
def test_dotted_extension_name(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, "pkg.mod")
    sys.path.insert(0, str(tmp_path))
    sys.modules.pop("pkg", None)
    sys.modules.pop("pkg.mod", None)
    try:
        mod = importlib.import_module("pkg.mod")
        assert mod.hello() == 7
        assert mod.__name__ == "pkg.mod"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("pkg.mod", None)
        sys.modules.pop("pkg", None)
