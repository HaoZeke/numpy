import shutil
import sys
from pathlib import Path

import pytest

from numpy.f2py._backends._meson import MesonTemplate

from . import util


def _meson_text():
    template = MesonTemplate(
        "m",
        [Path("a.f90")],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        "release",
        sys.executable,
    )
    return template.generate_meson_build()


def test_extension_links_with_the_fortran_compiler():
    # gh-25159: a C link of an ifx object cannot find libifcore.
    assert "link_language: 'fortran'" in _meson_text()


def test_ifx_runtime_imports(monkeypatch):
    # The missing libraries are ifx's. gfortran already finds its runtime
    # when the link is done by gcc, so this check is ifx-only.
    if shutil.which("ifx") is None:
        pytest.skip("ifx is not installed")
    ifx = Path(shutil.which("ifx")).resolve()
    lib = ifx.parents[1] / "lib"
    monkeypatch.setenv("FC", "ifx")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(lib))
    module = util.build_module(
        [util.getpath("tests", "src", "regression", "gh25159.f90")],
        module_name="gh25159_ifx",
    )
    module.sayhello()
