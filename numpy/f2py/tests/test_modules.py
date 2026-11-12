import textwrap

import pytest

import numpy as np

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
class TestModuleAttributes(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "module_data_docstring.f90")]

    def test_attribute_dict(self):
        mod = self.module.mod
        # __doc__ is built on first access and cached in the object's dict
        assert mod.__doc__ is mod.__doc__
        mod.custom = 42
        assert mod.custom == 42
        del mod.custom
        with pytest.raises(AttributeError):
            mod.custom
        with pytest.raises(AttributeError):
            del mod.custom
        with pytest.raises(AttributeError):
            mod.no_such_attribute


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
class TestAllocatableOnlyModule(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "modules", "gh25777_alloc_only.f90")
    ]

    def test_allocatable_only_module(self):
        # gh-25777: f2pyinit<mod> has no use statement for allocatables
        mod = self.module.alloc_only_mod
        assert mod.b is None
        mod.b = [[1, 2, 3], [4, 5, 6]]
        assert mod.b.flags.f_contiguous
        assert np.array_equal(mod.b, [[1, 2, 3], [4, 5, 6]])
        mod.b = None
        assert mod.b is None


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
class TestUsedParameters(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "gh3562.f90")]

    def test_gh3562(self):
        assert self.module.test.a == 1
        assert self.module.test.not_param == 5
        assert self.module.test2.b == 2
        assert self.module.test2.c == 1
        assert not hasattr(self.module.test2, "a")
        assert not hasattr(self.module.test2, "not_param")


def test_gh25777_allocatable_no_assumed_shape_warning(capfd):
    from numpy.f2py import f90mod_rules
    from numpy.f2py.crackfortran import crackfortran

    src = util.getpath("tests", "src", "modules", "gh25777.f90")
    blocks = crackfortran([str(src)])
    capfd.readouterr()
    f90mod_rules.buildhooks(blocks[0])
    out, err = capfd.readouterr()
    assert "assumed shape array" not in out + err
