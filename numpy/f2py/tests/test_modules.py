import textwrap

import pytest

from numpy.f2py._backends._meson import _prepare_objects, _prepare_sources

from . import util


@pytest.mark.slow
class TestModuleFilterPublicEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh26920",
            "two_mods_with_one_public_routine.f90"
        )
    ]
    # we filter the only public function mod2
    only = ["mod1_func1", ]

    def test_gh26920(self):
        # if it compiles and can be loaded, things are fine
        pass


@pytest.mark.slow
class TestModuleWithoutPublicEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh26920",
            "two_mods_with_no_public_entities.f90"
        )
    ]
    only = ["mod1_func1", ]

    def test_gh26920(self):
        # if it compiles and can be loaded, things are fine
        pass


@pytest.mark.slow
class TestModuleDocString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "module_data_docstring.f90")]

    def test_module_docstring(self):
        assert self.module.mod.__doc__ == textwrap.dedent(
            """\
                     i : 'i'-scalar
                     x : 'i'-array(4)
                     a : 'f'-array(2,3)
                     b : 'f'-array(-1,-1), not allocated
                     foo()\n
                     Wrapper for ``foo``.\n\n"""
        )


@pytest.mark.slow
class TestModuleAndSubroutine(util.F2PyTest):
    module_name = "example"
    sources = [
        util.getpath("tests", "src", "modules", "gh25337", "data.f90"),
        util.getpath("tests", "src", "modules", "gh25337", "use_data.f90"),
        util.getpath("tests", "src", "regression", "datonly.f90"),
    ]

    def test_gh25337(self):
        self.module.data.set_shift(3)
        assert "data" in dir(self.module)

    def test_allocatable_in_dir(self):
        # gh-27696: allocatable arrays should appear in dir()
        names = dir(self.module.datonly)
        assert "data_array" in names
        assert "max_value" in names


@pytest.mark.slow
class TestAllocatableCharacterArray(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "gh21674_alloc_char.f90")
    ]

    def test_allocatable_char_roundtrip(self):
        # gh-21674: allocatable character arrays should work after allocation
        mod = self.module.alloc_char_mod
        mod.setup_data(3)
        assert mod.charge.shape == (3, 3)
        names = mod.names
        assert names.shape == (3,)
        assert names.dtype.kind == 'U' or names.dtype.kind == 'S'


@pytest.mark.slow
class TestUsedModule(util.F2PyTest):
    module_name = "fmath"
    sources = [
        util.getpath("tests", "src", "modules", "use_modules.f90"),
    ]

    def test_gh25867(self):
        compiled_mods = [x for x in dir(self.module) if "__" not in x]
        assert "useops" in compiled_mods
        assert self.module.useops.sum_and_double(3, 7) == 20
        assert "mathops" in compiled_mods
        assert self.module.mathops.add(3, 7) == 10


class TestBuildDirIsSourceDir:
    # gh-29762: --build-dir equal to the source directory must not raise
    # SameFileError, and must not delete generated sources that already
    # live in that directory.
    def test_prepare_sources_keeps_files(self, tmp_path, monkeypatch):
        # f2py writes the generated sources into the cwd, which is the
        # build directory in this case.
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "addone.f90"
        src.write_text("subroutine addone\nend subroutine\n")
        generated = tmp_path / "addmodule.c"
        generated.write_text("int x;\n")
        library = tmp_path / "lib.a"
        library.write_bytes(b"archive")
        names = _prepare_sources(
            "add", [str(src), str(tmp_path / "skip.pyf")], tmp_path)
        _prepare_objects("add", [str(library)], tmp_path)
        assert src.read_text().startswith("subroutine")
        assert generated.read_text() == "int x;\n"
        assert library.read_bytes() == b"archive"
        assert "addone.f90" in names
        assert "addmodule.c" in names
        assert "skip.pyf" not in names

    def test_prepare_sources_still_copies(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        build_dir = tmp_path / "build"
        src_dir.mkdir()
        monkeypatch.chdir(src_dir)
        source = src_dir / "addone.f90"
        source.write_text("subroutine addone\nend subroutine\n")
        generated = src_dir / "addmodule.c"
        generated.write_text("int x;\n")
        names = _prepare_sources("add", [str(source)], build_dir)
        assert (build_dir / "addone.f90").read_text() == source.read_text()
        assert (build_dir / "addmodule.c").read_text() == "int x;\n"
        assert not generated.exists()
        assert names == ["addone.f90", "addmodule.c"]


@pytest.mark.slow
class TestBuildDirIsSourceDirCompile(util.F2PyTest):
    # gh-29762
    suffix = ".f90"
    code = """
subroutine addone(x, y)
  integer, intent(in) :: x
  integer, intent(out) :: y
  y = x + 1
end subroutine addone
"""
    options = ["--build-dir", util.get_module_dir()]

    def test_compiles_in_source_directory(self):
        assert self.module.addone(3) == 4
