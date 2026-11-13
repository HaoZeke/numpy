"""LFortran compiles f2py wrappers that use Fortran's implicit typing."""
import os

import pytest

from numpy.f2py._backends._meson import _lfortran_language_flags

from . import util


def test_lfortran_gets_implicit_typing(monkeypatch):
    monkeypatch.setenv(
        "FC", "/tmp/lfortran-src/lfortran/build/llvm11/src/bin/lfortran")
    flags = _lfortran_language_flags(["-O2"])
    assert flags == ["-O2", "--implicit-typing"]


def test_other_compilers_are_unchanged(monkeypatch):
    monkeypatch.setenv("FC", "/usr/bin/gfortran")
    assert _lfortran_language_flags(["-O2"]) == ["-O2"]
    monkeypatch.delenv("FC")
    assert _lfortran_language_flags([]) == []


@pytest.mark.skipif(
    "lfortran" not in os.environ.get("FC", ""),
    reason="needs FC pointing at LFortran",
)
def test_untyped_function_builds(tmp_path):
    # `function hi()` has no declared type. Fortran types it from the name.
    src = tmp_path / "hi.f90"
    src.write_text("function hi()\nhi = 1.0\nend function\n")
    mod = util.build_module([src], module_name="lf_implicit_hi")
    assert abs(mod.hi() - 1.0) < 1e-6
