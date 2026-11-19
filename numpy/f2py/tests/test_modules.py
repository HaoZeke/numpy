import textwrap

import numpy as np
import pytest
from numpy.testing import assert_array_equal

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
class TestSetattrRank(util.F2PyTest):
    suffix = ".f90"
    code = """
module m
  integer :: fixed(2, 2, 2) = 0
  integer, allocatable :: alloc(:,:,:)
end module m
"""

    @pytest.mark.parametrize(
        "attr, shape",
        [
            ("fixed", (1, 2, 2, 2)),
            ("alloc", (2, 2)),
        ],
    )
    def test_setattr_rank_mismatch(self, attr, shape):
        vals = np.arange(1, int(np.prod(shape)) + 1, dtype=np.int32)
        arr = vals.reshape(shape, order="F")
        setattr(self.module.m, attr, arr)
        got = getattr(self.module.m, attr)
        expected_shape = (2, 2, 2) if attr == "fixed" else (2, 2, 1)
        assert got.shape == expected_shape
        assert_array_equal(got, arr.reshape(expected_shape, order="F"))
