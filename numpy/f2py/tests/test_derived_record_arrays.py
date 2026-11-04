"""Native multidimensional, allocated, and pointer record components."""

import pytest

import numpy as np

from . import util


def records(atom, shape):
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = atom(x=1 + np.ravel_multi_index(index, shape))
    return result


def values(array):
    result = np.empty(array.shape)
    for index in np.ndindex(array.shape):
        result[index] = array[index].x
    return result


@pytest.mark.slow
class TestDerivedRecordArrays(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "record_array_boundaries.f90")]

    @pytest.mark.parametrize("representation", ["c", "o"])
    def test_fortran_coordinates(self, representation):
        grid = getattr(self.module, "record_grid_" + representation)()
        getattr(self.module.record_array_boundaries,
                "fill_grid_" + representation)(grid)
        np.testing.assert_array_equal(values(grid.plane), np.fromfunction(
            lambda i, j: 100 * (i + 1) + j + 1, (2, 3)))
        np.testing.assert_array_equal(values(grid.cube), np.fromfunction(
            lambda i, j, k: 100 * (i + 1) + 10 * (j + 1) + k + 1,
            (2, 2, 3)))
        assert grid.plane.flags.f_contiguous
        assert grid.cube.flags.f_contiguous

    @pytest.mark.parametrize("representation", ["c", "o"])
    def test_assignment_reaches_fortran(self, representation):
        grid = getattr(self.module, "record_grid_" + representation)()
        atom = getattr(self.module, "record_atom_" + representation)
        plane = records(atom, (2, 3))
        cube = records(atom, (2, 2, 3))
        grid.plane = plane
        grid.cube = cube
        read = getattr(self.module.record_array_boundaries,
                       "read_grid_" + representation)
        assert read(grid) == plane[0, 1].x + 10 * plane[1, 2].x + cube[1, 0, 2].x
        copied = grid.plane
        copied[0, 0].x = 1234
        assert grid.plane[0, 0].x == plane[0, 0].x

    @pytest.mark.parametrize("representation", ["c", "o"])
    def test_wrong_shape_and_type_preserve_components(self, representation):
        grid = getattr(self.module, "record_grid_" + representation)()
        atom = getattr(self.module, "record_atom_" + representation)
        grid.plane = records(atom, (2, 3))
        expected = values(grid.plane)
        with pytest.raises(ValueError):
            grid.plane = records(atom, (3, 2))
        invalid = records(atom, (2, 3))
        invalid[1, 2] = object()
        with pytest.raises(TypeError):
            grid.plane = invalid
        np.testing.assert_array_equal(values(grid.plane), expected)

    @pytest.mark.parametrize("shape", [(2, 3), (1, 4), (0, 3)])
    def test_allocated_shapes_and_none(self, shape):
        grid = self.module.record_allocated()
        assert grid.plane is None
        data = records(self.module.record_atom_o, shape)
        grid.plane = data
        assert grid.plane.shape == shape
        np.testing.assert_array_equal(values(grid.plane), values(data))
        grid.plane = None
        assert grid.plane is None

    def test_allocated_rank_three_and_interoperable_elements(self):
        grid = self.module.record_allocated()
        cube = records(self.module.record_atom_o, (2, 3, 4))
        grid.cube = cube
        np.testing.assert_array_equal(values(grid.cube), values(cube))
        plane = records(self.module.record_atom_c, (3, 2))
        grid.interoperable = plane
        np.testing.assert_array_equal(values(grid.interoperable), values(plane))
        grid.interoperable = records(self.module.record_atom_c, (1, 4))
        assert grid.interoperable.shape == (1, 4)

    @pytest.mark.parametrize("representation", ["c", "o"])
    def test_conversion_can_reinitialize_the_destination(self, representation):
        grid = getattr(self.module, "record_grid_" + representation)()
        atom = getattr(self.module, "record_atom_" + representation)
        expected = records(atom, (2, 3))

        class Producer:
            def __array__(self, dtype=None, copy=None):
                grid.__init__()
                return np.asarray(expected, dtype=dtype)

        grid.plane = Producer()
        np.testing.assert_array_equal(values(grid.plane), values(expected))

    def test_interoperable_scalar_copy_owns_its_storage(self):
        grid = self.module.record_allocated()
        grid.scalar = self.module.record_atom_c(x=47)
        copied = grid.scalar
        grid.scalar = self.module.record_atom_c(x=91)
        assert copied.x == 47
        del grid
        assert copied.x == 47
        copied.x = 23
        assert copied.x == 23

    def test_pointer_section_uses_logical_indices(self):
        pointer = self.module.record_pointer()
        assert pointer.plane is None
        self.module.record_array_boundaries.attach_records(pointer)
        copied = pointer.plane
        np.testing.assert_array_equal(values(copied), [[102, 104], [302, 304]])
        with pytest.raises(AttributeError, match="read-only"):
            pointer.plane = copied
        self.module.record_array_boundaries.detach_records(pointer)
        assert pointer.plane is None
        np.testing.assert_array_equal(values(copied), [[102, 104], [302, 304]])

    def test_nested_interoperability_uses_opaque_accessors(self):
        grid = self.module.record_char_grid()
        self.module.record_array_boundaries.fill_char_grid(grid)
        plane = grid.plane
        assert plane.shape == (2, 3)
        assert plane[1, 2].n == 203
        assert plane[1, 2].mark == "x"

    def test_fixed_zero_extent(self):
        grid = self.module.record_zero(tag=5)
        assert grid.plane.shape == (0, 2)
        grid.plane = np.empty((0, 2), dtype=object)
        assert grid.plane.shape == (0, 2)
        assert grid.tag == 5
        with pytest.raises(ValueError):
            grid.plane = np.empty((2, 0), dtype=object)

    def test_getters_use_intrinsic_copy_not_defined_assignment(self):
        box = self.module.intrinsic_copy_box()
        self.module.record_array_boundaries.fill_intrinsic_copy_box(box)
        scalar = box.scalar
        copied = box.items
        assert scalar.x == 42
        assert [item.x for item in copied] == [1, 2, 3]
        scalar.x = -1
        copied[0].x = -2
        assert box.scalar.x == 42
        assert box.items[0].x == 1

    def test_record_extent_exceeds_native_index_range(self):
        grid = self.module.record_allocated()
        extent = np.iinfo(np.intc).max + 1
        with pytest.raises(OverflowError):
            grid.plane = np.empty((0, extent), dtype=object)
        assert grid.plane is None

    def test_len_record_component_shape_is_queried_from_fortran(self):
        grid = self.module.record_len(n=2)
        expected = records(self.module.record_atom_o, (2, 2))
        grid.plane = expected
        np.testing.assert_array_equal(values(grid.plane), values(expected))
        with pytest.raises(ValueError):
            grid.plane = records(self.module.record_atom_o, (1, 2))
        empty = self.module.record_len(n=-1)
        assert empty.plane.shape == (0, 2)
        empty.plane = np.empty((0, 2), dtype=object)


@pytest.mark.slow
class TestDerivedKindArrays(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "kind_array_boundaries.f90")]

    @pytest.mark.parametrize("kind,dtype", [(4, "float32"), (8, "float64")])
    def test_fixed_and_allocated_members(self, kind, dtype):
        array = self.module.kind_array(k=kind)
        plane = np.arange(6, dtype=dtype).reshape(2, 3)
        array.plane = plane
        assert array.plane.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(array.plane, plane)
        cube = np.arange(24, dtype=dtype).reshape(2, 3, 4)
        assert array.cube is None
        array.cube = cube
        assert array.cube.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(array.cube, cube)
        array.cube = None
        assert array.cube is None
