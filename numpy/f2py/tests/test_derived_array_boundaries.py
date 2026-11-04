"""Native shape, layout, and storage lifetimes of derived arrays."""

import gc

import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestDerivedArrayBoundaries(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "array_boundaries.f90")]

    @pytest.mark.parametrize("representation", ["bindc", "opaque"])
    def test_fortran_coordinates(self, representation):
        p = getattr(self.module, "boundary_" + representation)()
        f = self.module.array_boundary_types
        getattr(f, "fill_" + representation)(p)
        expected_plane = np.fromfunction(
            lambda i, j: 100 * (i + 1) + j + 1, (2, 3))
        expected_cube = np.fromfunction(
            lambda i, j, k: 100 * (i + 1) + 10 * (j + 1) + k + 1,
            (2, 3, 4))
        np.testing.assert_array_equal(p.plane, expected_plane)
        np.testing.assert_array_equal(p.cube, expected_cube)
        assert p.plane.flags.f_contiguous
        assert p.cube.flags.f_contiguous

    @pytest.mark.parametrize("representation", ["bindc", "opaque"])
    @pytest.mark.parametrize("order", ["C", "F", "strided"])
    def test_assignment_is_read_by_fortran(self, representation, order):
        p = getattr(self.module, "boundary_" + representation)()
        plane = np.arange(6, dtype=np.float64).reshape(2, 3) + 1
        cube = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 20
        if order == "strided":
            plane = np.repeat(plane, 2, axis=1)[:, ::2]
            cube = np.repeat(cube, 2, axis=2)[:, :, ::2]
        else:
            plane = np.array(plane, order=order)
            cube = np.array(cube, order=order)
        p.plane = plane
        p.cube = cube
        f = getattr(self.module.array_boundary_types, "read_" + representation)
        assert f(p) == plane[0, 1] + 10 * plane[1, 2] + cube[1, 0, 3]

    @pytest.mark.parametrize("representation", ["bindc", "opaque"])
    def test_fixed_shape_is_checked(self, representation):
        p = getattr(self.module, "boundary_" + representation)()
        for value in (1.0, np.arange(6), np.zeros((3, 2))):
            with pytest.raises(ValueError):
                p.plane = value
        np.testing.assert_array_equal(p.plane, np.zeros((2, 3)))

    def test_view_keeps_its_allocation_across_reinitialization(self):
        p = self.module.boundary_bindc()
        p.plane = np.arange(6).reshape(2, 3)
        view = p.plane
        p.__init__()
        gc.collect()
        np.testing.assert_array_equal(view, np.arange(6).reshape(2, 3))
        view[1, 2] = 91
        assert view[1, 2] == 91
        np.testing.assert_array_equal(p.plane, np.zeros((2, 3)))
        del p
        gc.collect()
        assert view[1, 2] == 91

    @pytest.mark.parametrize("representation", ["bindc", "opaque"])
    def test_conversion_can_reinitialize_the_destination(self, representation):
        p = getattr(self.module, "boundary_" + representation)()
        expected = np.arange(6, dtype=np.float64).reshape(2, 3)

        class Producer:
            def __array__(self, dtype=None, copy=None):
                p.__init__()
                return np.asarray(expected, dtype=dtype)

        p.plane = Producer()
        np.testing.assert_array_equal(p.plane, expected)

    def test_temporary_array_entries_live_through_the_call(self):
        released = []
        observed = []
        atom_type = self.module.boundary_atom

        class Atom(atom_type):
            def __del__(self):
                released.append(True)

        class Producer:
            def __array__(self, dtype=None, copy=None):
                result = np.empty(2, dtype=object)
                result[:] = [Atom(x=3), Atom(x=7)]
                return result

        def observe(value):
            assert released == []
            observed.append(True)
            return value

        result = self.module.array_boundary_types.inspect_atoms(Producer(), observe)
        assert result == 10
        assert observed == [True]
        assert released == [True, True]

    def test_numeric_pointer_section_uses_logical_indices(self):
        pointer = self.module.boundary_pointer()
        assert pointer.plane is None
        self.module.array_boundary_types.attach_numeric_section(pointer)
        np.testing.assert_array_equal(pointer.plane, [[102, 104], [302, 304]])

    def test_fixed_zero_extent(self):
        value = self.module.boundary_zero(tag=7)
        assert value.plane.shape == (0, 3)
        value.plane = np.empty((0, 3))
        assert value.plane.shape == (0, 3)
        assert value.tag == 7
        with pytest.raises(ValueError):
            value.plane = np.empty((3, 0))

    def test_numeric_extent_exceeds_native_index_range(self):
        value = self.module.boundary_opaque()
        extent = np.iinfo(np.intc).max + 1
        with pytest.raises(OverflowError):
            value.dynamic_plane = np.empty((0, extent), dtype=np.float64)
        assert value.dynamic_plane is None

    def test_callback_can_release_the_source_container(self):
        released = []
        atom_type = self.module.boundary_atom

        class Atom(atom_type):
            def __del__(self):
                released.append(True)

        atoms = [Atom(x=3), Atom(x=7)]

        def observe(value):
            atoms.clear()
            assert released == []
            return value

        result = self.module.array_boundary_types.inspect_atoms(atoms, observe)
        assert result == 10
        assert released == [True, True]


@pytest.mark.slow
class TestDerivedLenArrayBoundaries(util.F2PyTest):
    sources = TestDerivedArrayBoundaries.sources

    def test_exact_len_shape_is_checked(self):
        p = self.module.boundary_len(n=2)
        expected = np.arange(6, dtype=np.float64).reshape(2, 3)
        p.plane = expected
        np.testing.assert_array_equal(p.plane, expected)
        for value in (1.0, np.arange(6), np.zeros((3, 2)), np.zeros((2, 2))):
            with pytest.raises(ValueError):
                p.plane = value
            np.testing.assert_array_equal(p.plane, expected)

    def test_len_expression_shape_is_queried_from_fortran(self):
        p = self.module.boundary_len(n=2)
        expected = np.arange(9, dtype=np.float64).reshape(3, 3)
        p.offset_plane = expected
        np.testing.assert_array_equal(p.offset_plane, expected)
        with pytest.raises(ValueError):
            p.offset_plane = np.zeros((2, 3))

    def test_negative_len_gives_zero_extent(self):
        p = self.module.boundary_len(n=-1)
        assert p.n == -1
        assert p.plane.shape == (0, 3)
        assert p.offset_plane.shape == (0, 3)
        p.plane = np.empty((0, 3))
        with pytest.raises(ValueError):
            p.plane = np.zeros((1, 3))
