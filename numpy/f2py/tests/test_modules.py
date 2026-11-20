import textwrap

import pytest

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


@pytest.mark.slow
class TestUsedCharacterLength(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "use_charlen", "user.f90"),
        util.getpath("tests", "src", "modules", "use_charlen", "nested.f90"),
        util.getpath("tests", "src", "modules", "use_charlen", "renamed.f90"),
        util.getpath("tests", "src", "modules", "use_charlen", "params.f90"),
    ]

    @pytest.mark.parametrize(
        "modname,attr,value",
        [
            ("u", "name", b"hello"),
            ("u", "tiny", b"ab"),
            ("u", "doubled", b"abcd"),
            ("u", "scaled", b"wxyz"),
            ("u", "starred", b"world"),
            ("ur", "name", b"hello"),
            ("ur_swap", "long_s", b"hello"),
            ("ur_swap", "short_s", b"ab"),
            ("ur_twice", "s", b"hello"),
            ("ur_twice", "t", b"ab"),
            ("ur_collide", "first", b"abc"),
            ("ur_collide_rev", "last", b"abc"),
            ("u_nested", "reexported", b"hello"),
            ("u_nested", "derived", b"0123456789"),
        ],
    )
    def test_module_variable_length(self, modname, attr, value):
        got = getattr(getattr(self.module, modname), attr)
        assert got.dtype.itemsize == len(value)
        assert got.tobytes() == value

    def test_contained_subroutine_length(self):
        assert self.module.u.get_name() == b"hello"
        assert self.module.u.echo(2) == b"ab"
