"""Native arrays of record components resolved through USE association."""

import pytest

import numpy as np
from numpy.f2py import crackfortran, derived_type_rules
from numpy.f2py.auxfuncs import F2PYError

from . import util
from .test_derived_record_arrays import records, values


def test_class_component_array_requires_dynamic_dispatch(tmp_path):
    source = tmp_path / "class_components.f90"
    source.write_text("""
module class_components
  type :: base
    real :: x
  end type base
  type, extends(base) :: child
    real :: y
  end type child
  type :: storage
    class(base), allocatable :: items(:)
  end type storage
end module class_components
""")
    parsed = crackfortran.crackfortran([str(source)])
    with pytest.raises(F2PYError, match="CLASS component array storage%items"):
        derived_type_rules.buildhooks(parsed[0])


@pytest.mark.slow
class TestImportedDerivedComponents(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types",
                            "imported_record_components.f90")]

    @pytest.mark.parametrize("representation", ["c", "o"])
    def test_renamed_private_alias_coordinates(self, representation):
        grid = getattr(self.module, "imported_" + representation + "_grid")()
        f = self.module.imported_record_components
        getattr(f, "fill_imported_" + representation)(grid)
        np.testing.assert_array_equal(values(grid.plane), np.fromfunction(
            lambda i, j: 100 * (i + 1) + j + 1, (2, 3)))
        np.testing.assert_array_equal(values(grid.cube), np.fromfunction(
            lambda i, j, k: 100 * (i + 1) + 10 * (j + 1) + k + 1,
            (2, 2, 3)))

    @pytest.mark.parametrize("representation", ["c", "o"])
    def test_renamed_alias_assignment_reaches_fortran(self, representation):
        grid = getattr(self.module, "imported_" + representation + "_grid")()
        atom = getattr(self.module, "record_source_" + representation).item
        plane = records(atom, (2, 3))
        cube = records(atom, (2, 2, 3))
        grid.plane = plane
        grid.cube = cube
        read = getattr(self.module.imported_record_components,
                       "read_imported_" + representation)
        assert read(grid) == plane[0, 1].x + 10 * plane[1, 2].x + cube[1, 0, 2].x

    def test_allocated_imported_record_components(self):
        grid = self.module.imported_alloc_grid()
        assert grid.plane is None
        assert grid.cube is None
        plane = records(self.module.record_source_o.item, (2, 3))
        cube = records(self.module.record_source_c.item, (2, 3, 4))
        grid.plane = plane
        grid.cube = cube
        np.testing.assert_array_equal(values(grid.plane), values(plane))
        np.testing.assert_array_equal(values(grid.cube), values(cube))
        grid.plane = records(self.module.record_source_o.item, (0, 3))
        assert grid.plane.shape == (0, 3)
        grid.cube = None
        assert grid.cube is None

    def test_use_only_retains_the_declaring_type(self):
        grid = self.module.only_record_grid()
        cube = records(self.module.record_source_c.item, (2, 3, 4))
        grid.cube = cube
        np.testing.assert_array_equal(values(grid.cube), values(cube))
        assert type(grid.cube[1, 2, 3]) is self.module.record_source_c.item
        assert self.module.imported_record_only.item is self.module.record_source_c.item

    def test_transitive_renamed_component(self):
        grid = self.module.transitive_record_grid()
        plane = records(self.module.record_source_o.item, (2, 3))
        grid.plane = plane
        np.testing.assert_array_equal(values(grid.plane), values(plane))
        assert type(grid.plane[1, 2]) is self.module.record_source_o.item
        assert (self.module.imported_record_reexport.renamed_item
                is self.module.record_source_o.item)
        assert (self.module.imported_record_transitive.local_item
                is self.module.record_source_o.item)

    def test_same_native_name_does_not_mix_representations(self):
        grid = self.module.imported_c_grid()
        wrong = records(self.module.record_source_o.item, (2, 3))
        with pytest.raises(TypeError):
            grid.plane = wrong
        assert self.module.record_source_c.item is not self.module.record_source_o.item
        assert not hasattr(self.module.imported_record_components, "c_record")
        assert not hasattr(self.module.imported_record_components, "o_record")

    def test_nested_import_keeps_its_member_resolution_context(self):
        grid = self.module.nested_record_grid()
        initial = grid.plane
        assert initial[1, 0].records[0, 2].x == 0
        nested = np.empty((2, 2), dtype=object)
        for index in np.ndindex(nested.shape):
            item = self.module.nested_record_item()
            item.records = records(self.module.record_source_o.item, (2, 3))
            nested[index] = item
        grid.plane = nested
        assert self.module.imported_nested_consumer.read_nested(grid) == 3
