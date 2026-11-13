import textwrap

import pytest

from numpy.f2py import crackfortran

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


def test_module_subroutine_is_not_a_nested_module(tmp_path):
    src = tmp_path / "submod.f90"
    src.write_text(
        "module parent\n"
        "  interface\n"
        "    module subroutine s(n)\n"
        "      integer, intent(out) :: n\n"
        "    end subroutine\n"
        "  end interface\n"
        "end module\n"
        "submodule (parent) child\n"
        "contains\n"
        "  module subroutine s(n)\n"
        "    integer, intent(out) :: n\n"
        "    n = 4\n"
        "  end subroutine\n"
        "end submodule\n"
    )
    blocks = crackfortran.crackfortran([str(src)])
    modules = [b for b in blocks if b.get("block") == "module" and b.get("name") == "parent"]
    routines = []
    for mod in modules:
        for body in mod.get("body", []):
            if body.get("block") == "subroutine" and body.get("name") == "s":
                routines.append(body)
    assert routines


@pytest.mark.slow
class TestSubmoduleProcedure(util.F2PyTest):
    suffix = ".f90"
    code = """
module parent
  interface
    module subroutine s(n)
      integer, intent(out) :: n
    end subroutine
  end interface
end module

submodule (parent) child
contains
  module subroutine s(n)
    integer, intent(out) :: n
    n = 4
  end subroutine
end submodule
"""

    def test_submodule_procedure(self):
        assert self.module.parent.s() == 4


@pytest.mark.slow
class TestInterfaceOnlyModule(util.F2PyTest):
    # The interface copy of a module has no routine. It still has to
    # import, and a later submodule implementation of the same name
    # must not be dropped.
    suffix = ".f90"
    code = """
module parent
  interface
    subroutine s(n)
      integer, intent(out) :: n
    end subroutine
  end interface
end module
"""

    def test_import_exposes_module(self):
        assert hasattr(self.module, "parent")


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
