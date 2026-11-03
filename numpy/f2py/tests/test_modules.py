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
class TestModuleWithoutWrappableEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh20053", "no_wrappable_entities.f90"
        )
    ]

    def test_gh20053(self):
        # Importing used to crash on the empty tables of these modules
        assert self.module.twice(2.0) == 4.0
        assert not hasattr(self.module, "type_only")
        assert not hasattr(self.module, "interface_only")


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
class TestModuleDataDelete(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "module_data_docstring.f90")]

    @pytest.mark.parametrize("name", ["i", "x", "a", "b"])
    def test_delete_module_data(self, name):
        mod = self.module.mod
        mod.b = [[1.0, 2.0, 3.0]]
        with pytest.raises(AttributeError, match="cannot delete fortran data"):
            delattr(mod, name)
        assert getattr(mod, name) is not None


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
class TestStrictSetattr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "gh3751.f90")]
    options = ["--strict-setattr"]

    def test_wrapped_names_are_assignable(self):
        mod = self.module.gh3751_mod
        mod.counter = 3
        assert mod.counter == 3
        mod.values = [1.0, 2.0]
        assert mod.values_size() == 2
        mod.values = None
        assert mod.values_size() == -1
        self.module.gh3751_cb.n = 4
        assert self.module.gh3751_cb.n == 4

    @pytest.mark.parametrize("name", ["hidden", "value"])
    def test_module_rejects_unwrapped_name(self, name):
        mod = self.module.gh3751_mod
        with pytest.raises(AttributeError, match=f"no attribute '{name}'"):
            setattr(mod, name, [1.0, 2.0])
        with pytest.raises(AttributeError, match=f"no attribute '{name}'"):
            delattr(mod, name)
        assert name not in dir(mod)

    def test_common_block_rejects_unwrapped_name(self):
        with pytest.raises(AttributeError, match="no attribute 'm'"):
            self.module.gh3751_cb.m = 1

    def test_routine_accepts_attributes(self):
        func = self.module.gh3751_mod.values_size
        func.tag = "kept"
        assert func.tag == "kept"


@pytest.mark.slow
class TestDefaultSetattr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "gh3751.f90")]

    def test_module_stores_unwrapped_name(self):
        mod = self.module.gh3751_mod
        mod.hidden = [1.0, 2.0]
        assert mod.hidden == [1.0, 2.0]
        assert mod.hidden_size() == -1
