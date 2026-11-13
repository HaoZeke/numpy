import sys
import textwrap

import pytest

from numpy.f2py._backends._meson import MesonTemplate, _macro_flags

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


def test_define_macros_are_compiler_args():
    # gh-28648
    flags = _macro_flags([("NEW_MODEL_PARM", None), ("LEVEL", "2")], ["OLD"])
    assert flags == ["-DNEW_MODEL_PARM", "-DLEVEL=2", "-UOLD"]
    text = MesonTemplate(
        modulename="model",
        sources=[],
        deps=[],
        libraries=[],
        library_dirs=[],
        include_dirs=[],
        object_files=[],
        linker_args=[],
        fortran_args=flags,
        c_args=flags,
        build_type="release",
        python_exe=sys.executable,
    ).generate_meson_build()
    assert "'-DNEW_MODEL_PARM'" in text
    assert "'-DLEVEL=2'" in text
    assert "'-UOLD'" in text
    assert "fortran_args:" in text
    assert "c_args:" in text


@pytest.mark.slow
class TestDefineMacro(util.F2PyTest):
    # gh-28648: -D must reach gfortran or the #ifdef routine is missing.
    suffix = ".f90"
    options = ["-DNEW_MODEL_PARM", "--f90flags=-cpp"]
    code = """
module model
  implicit none
contains
  subroutine run_passive_model(x, y)
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    y = 20 * x
  end subroutine run_passive_model
#ifdef NEW_MODEL_PARM
  subroutine run_active_model(x, y)
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    y = 10 * x
  end subroutine run_active_model
#endif
end module model
"""

    def test_ifdef_subroutine(self):
        assert self.module.model.run_passive_model(2.0) == 40.0
        assert self.module.model.run_active_model(2.0) == 20.0
