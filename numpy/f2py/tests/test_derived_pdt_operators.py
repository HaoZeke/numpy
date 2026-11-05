"""Native operators use owned descriptors and actual module selections."""

import pytest

import numpy as np
from numpy.f2py.tests import util


@pytest.mark.slow
class TestDerivedPDTOperators(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_operator_selectors.f90')]

    def values(self):
        module = self.module.pdt_operator_values
        module.reset_values()
        return module

    @pytest.mark.parametrize('left_module,right_module',
                             [(False, False), (True, False),
                              (False, True), (True, True)])
    def test_owned_and_selected_binary_operands(self, left_module, right_module):
        module = self.values()
        left = module.single if left_module else module.op_cell(k=4, value=3)
        right = (module.fixed[0, 0] if right_module
                 else module.op_cell(k=4, value=5))
        result = left + right
        assert result.value == (7 if left_module else 3) + (9 if right_module else 5)
        assert result.k == 4
        result.value = 101
        assert left.value == (7 if left_module else 3)
        assert right.value == (9 if right_module else 5)

    def test_unary_operations_copy_selected_storage(self):
        module = self.values()
        assert (+module.single).value == 7
        assert (-module.single).value == -7
        assert module.single.value == 7

    def test_same_spelled_types_keep_their_native_operator_owner(self):
        first = self.values()
        second = self.module.pdt_operator_other
        second.reset_values()
        assert (-first.single).value == -7
        assert (-second.single).value == -31
        assert first.single.value == 7
        assert second.single.value == 31

    def test_integer_and_reflected_operands(self):
        module = self.values()
        assert (module.single + 3).value == 10
        assert (3 + module.single).value == 10
        assert module.single.value == 7

    def test_mixed_kind_result_uses_its_canonical_specialization(self):
        module = self.values()
        result = module.single + module.wide
        assert result.k == module.wide.k
        assert result.value == 26
        assert result is not module.wide

    def test_logical_and_integer_results_are_native_results(self):
        module = self.values()
        assert module.single == module.op_cell(k=4, value=7)
        assert not (module.single == module.fixed[0, 0])
        assert module.single * module.fixed[0, 0] == 63

    def test_assignment_reaches_selected_destination(self):
        module = self.values()
        destination = module.fixed[0, 0]
        assert destination.__fortran_assign__(module.single) is destination
        assert module.fixed[0, 0].value == 57
        assert module.single.value == 7

    def test_reexport_self_assignment_snapshots_rhs(self):
        module = self.values()
        destination = module.single
        source = self.module.pdt_operator_facade.renamed_single
        assert destination.__fortran_assign__(source) is destination
        assert module.single.value == 57
        assert source.value == 57

    def test_target_pointer_alias_assignment_snapshots_rhs(self):
        module = self.values()
        destination = module.target_cell
        assert destination.__fortran_assign__(module.pointer_cell) is destination
        assert module.target_cell.value == 61
        assert module.pointer_cell.value == 61

    def test_pointer_operands_use_real_target_or_pointer_actuals(self):
        module = self.values()
        assert module.target_cell / module.pointer_cell == 1
        value = module.op_cell(k=4, value=22)
        assert value / module.target_cell == 2
        with pytest.raises((TypeError, ValueError)):
            module.single / module.target_cell
        assert module.single.value == 7

    def test_protected_operand_is_readable_but_not_assignable(self):
        module = self.values()
        assert (module.locked + module.single).value == 30
        with pytest.raises((TypeError, ValueError, AttributeError)):
            module.locked.__fortran_assign__(module.single)
        assert module.locked.value == 23

    def test_conversion_reinitialization_precedes_handle_extraction(self):
        value = self.module.pdt_operator_values.op_cell(k=4, value=3)

        class Reinitialize:
            def __index__(self):
                value.__init__(k=4, value=20)
                return 4

        result = value + Reinitialize()
        assert result.value == 24
        assert value.value == 20

    @pytest.mark.parametrize('length', [-3, 0, 2, 3])
    def test_signed_and_positive_result_lengths(self, length):
        source = self.module.len_cell(n=length, value=5)
        source.values = np.arange(max(length, 0), dtype=np.int32)
        result = -source
        assert result.n == length
        assert result.value == -5
        np.testing.assert_array_equal(result.values, -source.values)

    def test_fixed_len_operand_rejects_mismatch_without_mutation(self):
        source = self.module.len_cell(n=3, value=5)
        source.values = [1, 2, 3]
        with pytest.raises((ValueError, TypeError)):
            source + 4
        assert source.n == 3
        assert source.value == 5
        np.testing.assert_array_equal(source.values, [1, 2, 3])

    def test_len_selector_result_is_owned(self):
        module = self.module.pdt_operator_lengths
        module.reset_values()
        result = -module.single
        assert result.n == 2
        assert result.value == -7
        np.testing.assert_array_equal(result.values, [-8, -9])
        result.value = 99
        assert module.single.value == 7

    def test_allocatable_actual_keeps_fixed_len_mode(self):
        source = self.module.len_cell(n=2, value=3)
        source.values = [4, 5]
        assert source == 12
        mismatch = self.module.len_cell(n=3, value=3)
        mismatch.values = [4, 5, 6]
        with pytest.raises((ValueError, TypeError)):
            mismatch == 12
        assert mismatch.n == 3
        np.testing.assert_array_equal(mismatch.values, [4, 5, 6])

    def test_assumed_pointer_len_uses_checked_block_local_alias(self):
        module = self.module.pdt_operator_lengths
        module.reset_values()
        source = self.module.len_cell(n=2, value=3)
        source.values = [4, 5]
        assert source * 2 == 14
        assert module.target_cell * 2 == 26
        assert module.target_cell.n == 2

    def test_deferred_pointer_len_requires_matching_selector_mode(self):
        module = self.module.pdt_operator_lengths
        module.reset_values()
        source = self.module.len_cell(n=2, value=3)
        source.values = [4, 5]
        assert source / 2 == 14
        assert module.deferred_pointer / 2 == 26
        with pytest.raises((ValueError, TypeError)):
            module.target_cell / 2
        assert module.target_cell.n == 2

    @pytest.mark.parametrize('selected', [False, True])
    def test_allocatable_assignment_replaces_owned_or_actual_allocation(self, selected):
        module = self.module.pdt_operator_allocations
        module.reset_values()
        destination = (module.destination if selected
                       else self.module.allocation_cell(n=2, value=7))
        source = self.module.allocation_cell(n=2, value=11)
        source.values = [12, 13]
        assert destination.__fortran_assign__(source) is destination
        assert destination.n == 3
        assert destination.value == 61
        np.testing.assert_array_equal(destination.values, [12, 13, 99])
        assert source.n == 2
        assert source.value == 11
        np.testing.assert_array_equal(source.values, [12, 13])
        assert module.destination.n == (3 if selected else 2)

    def test_allocatable_self_assignment_preserves_rhs_before_reallocation(self):
        module = self.module.pdt_operator_allocations
        module.reset_values()
        destination = module.destination
        assert destination.__fortran_assign__(module.destination) is destination
        assert destination.n == 3
        assert destination.value == 57
        np.testing.assert_array_equal(destination.values, [8, 9, 99])

    def test_allocatable_input_is_allocated_for_owned_and_selected_values(self):
        module = self.module.pdt_operator_allocations
        module.reset_values()
        owned = self.module.allocation_cell(n=2, value=3)
        owned.values = [4, 5]
        assert owned * 2 == 14
        assert module.destination * 2 == 26
        assert module.destination.n == 2

    def test_allocatable_destination_rejects_nonallocatable_selector(self):
        module = self.module.pdt_operator_allocations
        module.reset_values()
        source = self.module.allocation_cell(n=2, value=11)
        source.values = [12, 13]
        with pytest.raises((ValueError, TypeError)):
            module.fixed.__fortran_assign__(source)
        assert module.fixed.n == 2
        assert module.fixed.value == 17
        np.testing.assert_array_equal(module.fixed.values, [18, 19])

    @pytest.mark.parametrize('selected', [False, True])
    def test_allocatable_deallocation_invalidates_existing_proxy(self, selected):
        module = self.module.pdt_operator_allocations
        module.reset_values()
        destination = (module.destination if selected
                       else self.module.allocation_cell(n=2, value=7))
        source = self.module.allocation_cell(n=2, value=-1)
        source.values = [12, 13]
        assert destination.__fortran_assign__(source) is destination
        if selected:
            assert module.destination is None
        for name in ('value', 'values', 'n'):
            with pytest.raises(ReferenceError):
                getattr(destination, name)
        with pytest.raises(ReferenceError):
            destination * 2
        assert source.value == -1
        assert source.n == 2

    def test_owned_deallocated_object_can_be_reinitialized(self):
        destination = self.module.allocation_cell(n=2, value=7)
        source = self.module.allocation_cell(n=2, value=-1)
        source.values = [12, 13]
        assert destination.__fortran_assign__(source) is destination
        destination.__init__(n=1, value=9)
        destination.values = [10]
        assert destination.n == 1
        assert destination.value == 9
        np.testing.assert_array_equal(destination.values, [10])
