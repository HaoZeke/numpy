import pytest

from numpy.f2py._backends._meson import _prepare_sources

from . import util


@pytest.mark.slow
class TestIncludedUse(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "incuse", "getn.f"),
        util.getpath("tests", "src", "modules", "incuse", "params.f90"),
    ]
    options = ["-I" + str(util.getpath("tests", "src", "modules", "incuse"))]

    def test_included_use(self):
        assert int(self.module.getn()) == 7


@pytest.mark.slow
class TestIncludedModule(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "incmod", "getn.f90"),
        util.getpath("tests", "src", "modules", "incmod", "host.f90"),
    ]
    options = ["-I" + str(util.getpath("tests", "src", "modules", "incmod"))]

    def test_included_module(self):
        assert int(self.module.getn()) == 11


@pytest.mark.slow
class TestNestedIncludedModule(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "incnest", "getn.f90"),
        util.getpath("tests", "src", "modules", "incnest", "host.f90"),
    ]
    options = ["-I" + str(util.getpath("tests", "src", "modules", "incnest"))]

    def test_nested_included_module(self):
        assert int(self.module.getn()) == 4


def test_nested_include_inlines_module(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (tmp_path / "types.inc").write_text("integer :: n = 1\n")
    (pkg / "defs.inc").write_text("include 'types.inc'\n")
    (pkg / "types.inc").write_text(
        "module params\n  integer :: n = 4\nend module\n"
    )
    src = tmp_path / "host.f90"
    original = "include 'pkg/defs.inc'\n"
    src.write_text(original)
    bdir = tmp_path / "build"
    _prepare_sources("m", [str(src)], bdir, [str(tmp_path)])
    # gfortran opens tmp_path/types.inc, which has no module or use.
    assert (bdir / "host.f90").read_text() == original


def test_nested_include_inlines_include_path_module(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (tmp_path / "types.inc").write_text(
        "module params\n  integer :: n = 4\nend module\n"
    )
    (pkg / "defs.inc").write_text("include 'types.inc'\n")
    (pkg / "types.inc").write_text(
        "module params\n  integer :: n = 9\nend module\n"
    )
    src = tmp_path / "host.f90"
    src.write_text("include 'pkg/defs.inc'\n")
    bdir = tmp_path / "build"
    _prepare_sources("m", [str(src)], bdir, [str(tmp_path)])
    text = (bdir / "host.f90").read_text()
    assert "module params" in text
    assert "n = 4" in text
    assert "n = 9" not in text


def test_plain_include_unchanged(tmp_path):
    src = tmp_path / "a.f90"
    original = "include 'plain.inc'\ninteger :: n = 3\n"
    src.write_text(original)
    (tmp_path / "plain.inc").write_text("integer, parameter :: k = 3\n")
    bdir = tmp_path / "build"
    _prepare_sources("m", [str(src)], bdir, [str(tmp_path)])
    assert (bdir / "a.f90").read_text() == original


def test_preprocessed_include_unchanged(tmp_path):
    src = tmp_path / "a.F90"
    original = "include 'defs.inc'\n"
    src.write_text(original)
    (tmp_path / "defs.inc").write_text(
        "module params\n  integer :: n = 1\nend module\n"
    )
    bdir = tmp_path / "build"
    _prepare_sources("m", [str(src)], bdir, [str(tmp_path)])
    assert (bdir / "a.F90").read_text() == original


def test_missing_include_unchanged(tmp_path):
    src = tmp_path / "a.f90"
    original = "include 'missing.inc'\n"
    src.write_text(original)
    bdir = tmp_path / "build"
    _prepare_sources("m", [str(src)], bdir, [str(tmp_path)])
    assert (bdir / "a.f90").read_text() == original
