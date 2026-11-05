"""Native module state uses validated parameterized selectors."""

import gc

import pytest

import numpy as np

from . import util


def field_values(array):
    result = np.empty(array.shape)
    for index in np.ndindex(array.shape):
        result[index] = array[index].value
    return result


@pytest.mark.slow
class TestDerivedModuleVariables(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_module_variables.f90')]

    def reset(self):
        module = self.module.pdt_module_values
        module.reset_values()
        return module

    def test_scalar_proxy_reaches_actual_non_target(self):
        module = self.reset()
        first, second = module.single, module.single
        assert first is not second
        assert first.value == 7
        first.value = 81
        assert second.value == 81
        assert module.scalar_value() == 81

    @pytest.mark.parametrize('name,shape', [('fixed', (2, 3)), ('cube', (2, 2, 3))])
    def test_rank_and_native_lower_bounds(self, name, shape):
        module = self.reset()
        result = getattr(module, name)
        assert result.shape == shape
        assert result.flags.f_contiguous
        for index in np.ndindex(shape):
            if name == 'fixed':
                expected = 100 * (index[0] + 1) + index[1] + 1
            else:
                expected = 100 * (index[0] + 1) + 10 * (index[1] + 1) + index[2] + 1
            assert result[index].value == expected
        result[(0,) * len(shape)].value = 301
        assert getattr(module, name)[(0,) * len(shape)].value == 301

    def test_record_assignment_and_native_readback(self):
        module = self.reset()
        module.single = self.module.cell(n=2, value=35)
        assert module.scalar_value() == 35
        values = module.fixed
        values[0, 0].value = 502
        module.fixed = values[:, ::-1]
        assert module.fixed_value(-1, 5) == 502

    def test_aliasing_assignment_copies_before_native_mutation(self):
        module = self.reset()
        expected = field_values(module.fixed)[:, ::-1].copy()
        module.fixed = module.fixed[:, ::-1]
        np.testing.assert_array_equal(field_values(module.fixed), expected)

    @pytest.mark.parametrize('bad', [np.empty((3, 2), dtype=object),
                                     np.empty((2, 3, 1), dtype=object)])
    def test_failed_shape_or_conversion_preserves_destination(self, bad):
        module = self.reset()
        expected = field_values(module.fixed)
        with pytest.raises((ValueError, TypeError)):
            module.fixed = bad
        np.testing.assert_array_equal(field_values(module.fixed), expected)

    def test_invalid_late_element_preserves_destination(self):
        module = self.reset()
        expected = field_values(module.fixed)
        incoming = module.fixed
        incoming[1, 2] = object()
        with pytest.raises(TypeError):
            module.fixed = incoming
        np.testing.assert_array_equal(field_values(module.fixed), expected)

    def test_typed_wrong_shape_preserves_destination(self):
        module = self.reset()
        expected = field_values(module.fixed)
        incoming = np.empty((3, 2), dtype=object)
        for index in np.ndindex(incoming.shape):
            incoming[index] = self.module.cell(n=2, value=99)
        with pytest.raises(ValueError):
            module.fixed = incoming
        np.testing.assert_array_equal(field_values(module.fixed), expected)

    def test_fixed_len_mismatch_preserves_destination(self):
        module = self.reset()
        with pytest.raises(ValueError):
            module.single = self.module.cell(n=3, value=99)
        assert module.scalar_value() == 7
        np.testing.assert_array_equal(module.single.values, [8, 9])
        module.associate_pool()
        incoming = np.empty((2, 3), dtype=object)
        for index in np.ndindex(incoming.shape):
            incoming[index] = self.module.cell(n=3, value=99)
        with pytest.raises(ValueError):
            module.selected = incoming
        np.testing.assert_array_equal(field_values(module.pool),
                                      field_values(module.fixed))

    def test_scalar_len_refresh_and_deallocation(self):
        module = self.reset()
        assert module.optional is None
        module.allocate_optional(2)
        held = module.optional
        assert held.values.shape == (2,)
        module.allocate_optional(5)
        assert held.values.shape == (5,)
        np.testing.assert_array_equal(held.values, np.full(5, 5))
        module.optional = None
        assert module.optional is None
        with pytest.raises(ReferenceError):
            _ = held.value

    def test_array_selection_revalidates_bounds(self):
        module = self.reset()
        assert module.allocated_cells is None
        module.resize_cells(2, 3, 2)
        selected = module.allocated_cells[1, 2]
        assert selected.value == 17
        module.resize_cells(1, 1, 4)
        with pytest.raises(ReferenceError):
            _ = selected.value
        assert module.allocated_cells[0, 0].values.shape == (4,)

    def test_allocated_assignment_preserves_input_selectors(self):
        module = self.reset()
        module.allocated_cells = module.fixed
        np.testing.assert_array_equal(field_values(module.allocated_cells),
                                      field_values(module.fixed))
        expected = field_values(module.allocated_cells)[:, ::-1].copy()
        module.allocated_cells = module.allocated_cells[:, ::-1]
        np.testing.assert_array_equal(field_values(module.allocated_cells), expected)
        module.allocated_cells = None
        assert module.allocated_cells is None

    def test_empty_allocated_array_keeps_declared_lengths(self):
        module = self.reset()
        module.resize_cells(1, 1, 3)
        module.allocated_cells = np.empty((0, 3), dtype=object)
        assert module.allocated_cells.shape == (0, 3)

    def test_fixed_len_allocatable_rejects_mismatch(self):
        module = self.reset()
        assert module.fixed_allocated is None
        module.fixed_allocated = module.fixed
        expected = field_values(module.fixed_allocated)
        incoming = np.empty((2, 3), dtype=object)
        for index in np.ndindex(incoming.shape):
            incoming[index] = self.module.cell(n=3, value=99)
        with pytest.raises(ValueError):
            module.fixed_allocated = incoming
        np.testing.assert_array_equal(field_values(module.fixed_allocated), expected)
        module.fixed_allocated = None
        module.fixed_allocated = np.empty((0, 2), dtype=object)
        assert module.fixed_allocated.shape == (0, 2)

    def test_pointer_selection_does_not_escape_non_target_storage(self):
        module = self.reset()
        assert module.selected is None
        module.associate_pool()
        held = module.selected[1, 2]
        held.value = 79
        assert module.pool[1, 2].value == 79
        module.selected = None
        assert module.selected is None
        assert module.pool[1, 2].value == 79
        with pytest.raises(ReferenceError):
            _ = held.value

    def test_protected_selection_refuses_mutation(self):
        module = self.reset()
        protected = module.locked
        assert protected.value == 23
        with pytest.raises((AttributeError, TypeError, ValueError)):
            protected.value = 18
        with pytest.raises(AttributeError):
            module.locked = self.module.cell(n=2)
        assert module.locked.value == 23
        assert not hasattr(module, 'hidden')

    def test_numeric_component_reads_are_owned_copies(self):
        module = self.reset()
        result = module.single.values
        result[:] = 909
        np.testing.assert_array_equal(module.single.values, [8, 9])

    def test_routine_argument_uses_the_actual_module_variable(self):
        module = self.reset()
        module.bump_cell(module.single, 11)
        assert module.scalar_value() == 18
        module.bump_cell(module.fixed[1, 2], 31)
        assert module.fixed_value(0, 5) == 234

    def test_protected_selection_is_a_legal_read_source(self):
        module = self.reset()
        module.single = module.locked
        assert module.scalar_value() == 23
        assert module.locked.value == 23

    def test_selector_retains_module_owner(self):
        module = self.reset()
        selected = module.single
        del module
        gc.collect()
        assert selected.value == 7
        selected.value = 63
        assert self.module.pdt_module_values.scalar_value() == 63

    def test_use_renamed_variable_preserves_owner_identity(self):
        module = self.reset()
        facade = self.module.pdt_module_facade
        assert facade.renamed_single.value == 7
        facade.renamed_single.value = 125
        assert module.scalar_value() == 125
        module.single = facade.renamed_single
        assert module.single.value == 125


@pytest.mark.slow
class TestDerivedModuleDefinedAssignment(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_module_assignment.f90')]

    def test_snapshot_uses_intrinsic_copy_and_assignment_runs_once(self):
        module = self.module.pdt_module_assignment
        module.reset_values()
        first = self.module.assigned_cell(n=2, value=7)
        second = self.module.assigned_cell(n=2, value=9)
        module.fixed = [first, second]
        np.testing.assert_array_equal(field_values(module.fixed), [57, 59])
        assert first.value == 7
        assert second.value == 9

    def test_self_aliasing_sources_are_snapshotted_intrinsically(self):
        module = self.module.pdt_module_assignment
        module.reset_values()
        module.fixed = module.fixed[::-1]
        np.testing.assert_array_equal(field_values(module.fixed), [52, 51])

    def test_allocatable_commit_does_not_repeat_defined_assignment(self):
        module = self.module.pdt_module_assignment
        module.reset_values()
        module.allocated_cells = module.fixed
        np.testing.assert_array_equal(field_values(module.allocated_cells), [51, 52])
        module.allocated_cells = module.allocated_cells[::-1]
        np.testing.assert_array_equal(field_values(module.allocated_cells), [102, 101])

    def test_public_generic_assignment_calls_private_specific_once(self):
        module = self.module.pdt_module_generic_assignment
        module.reset_values()
        first = self.module.generic_cell(n=2, value=7)
        second = self.module.generic_cell(n=2, value=9)
        module.fixed = [first, second]
        np.testing.assert_array_equal(field_values(module.fixed), [37, 39])
        module.fixed = module.fixed[::-1]
        np.testing.assert_array_equal(field_values(module.fixed), [69, 67])
        assert first.value == 7
        assert second.value == 9
