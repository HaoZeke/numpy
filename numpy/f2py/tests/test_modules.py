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
class TestPointerModuleArray(util.F2PyTest):
    suffix = ".f90"
    code = """
        module pointer_module
          logical(kind=1), pointer :: values(:)
          integer, pointer :: scalar
          integer, pointer :: locked(:)
          character(len=:), pointer :: name
          protected :: locked
          integer, target :: storage(6)
          integer, pointer :: view(:)
          integer, target :: cell
          integer, pointer :: cell_p
          integer, allocatable, target :: heap(:)
        contains
          subroutine associate_view()
            storage = [1, 2, 3, 4, 5, 6]
            view => storage
            cell = 5
            cell_p => cell
          end subroutine associate_view
          subroutine associate_stride()
            storage = [1, 2, 3, 4, 5, 6]
            view => storage(1:6:2)
          end subroutine associate_stride
          subroutine associate_section()
            storage = [1, 2, 3, 4, 5, 6]
            view => storage(2:5)
          end subroutine associate_section
          subroutine associate_empty()
            view => storage(1:0)
          end subroutine associate_empty
          subroutine free_view()
            deallocate(view)
          end subroutine free_view
          subroutine associate_heap()
            allocate(heap(6))
            heap = [1, 2, 3, 4, 5, 6]
            view => heap
          end subroutine associate_heap
        end module pointer_module
    """

    def test_array_pointer(self):
        mod = self.module.pointer_module
        assert mod.values is None
        mod.values = np.array([True, False, True])
        assert np.array_equal(mod.values, [1, 0, 1])
        with pytest.raises(ValueError, match="cannot be reallocated"):
            mod.values = True
        mod.values = None
        assert mod.values is None
        mod.values = True
        assert np.array_equal(mod.values, [1])

    def test_scalar_pointer(self):
        mod = self.module.pointer_module
        assert mod.scalar is None
        mod.scalar = 4
        assert mod.scalar == 4
        with pytest.raises(ValueError, match="too many axes"):
            mod.scalar = np.array([1, 2])
        mod.scalar = None
        assert mod.scalar is None

    def test_delete_raises(self):
        mod = self.module.pointer_module
        with pytest.raises(AttributeError):
            del mod.values

    def test_target_pointer(self):
        mod = self.module.pointer_module
        mod.associate_view()
        assert np.array_equal(mod.view, [1, 2, 3, 4, 5, 6])
        with pytest.raises(ValueError, match="cannot be reallocated"):
            mod.view = np.zeros(2, dtype=np.int32)
        mod.view = np.array([9, 8, 7, 6, 5, 4], dtype=np.int32)
        assert np.array_equal(mod.storage, [9, 8, 7, 6, 5, 4])
        mod.cell_p = 11
        assert mod.cell == 11
        mod.view = None
        mod.cell_p = None
        assert mod.view is None
        assert mod.cell_p is None
        assert np.array_equal(mod.storage, [9, 8, 7, 6, 5, 4])
        assert mod.cell == 11

    def test_freed_block_not_reused(self):
        mod = self.module.pointer_module
        mod.view = np.arange(6, dtype=np.int32)
        mod.free_view()
        mod.associate_heap()
        mod.view = None
        assert np.array_equal(mod.heap, [1, 2, 3, 4, 5, 6])

    def test_strided_pointer_raises(self):
        mod = self.module.pointer_module
        mod.associate_stride()
        with pytest.raises(ValueError, match="not contiguous"):
            mod.view
        with pytest.raises(ValueError, match="not contiguous"):
            mod.view = np.array([1, 3, 5], dtype=np.int32)
        mod.view = None
        assert mod.view is None
        assert np.array_equal(mod.storage, [1, 2, 3, 4, 5, 6])
        mod.associate_section()
        assert np.array_equal(mod.view, [2, 3, 4, 5])
        mod.view = np.array([7, 7, 7, 7], dtype=np.int32)
        assert np.array_equal(mod.storage, [1, 7, 7, 7, 7, 6])
        mod.associate_empty()
        assert mod.view.size == 0
        mod.view = None
        assert mod.view is None


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
