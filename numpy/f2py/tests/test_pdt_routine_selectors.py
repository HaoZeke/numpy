"""Routine calls preserve owned and module-backed parameterized values."""

import pytest

import numpy as np
from numpy.f2py import crackfortran, derived_type_rules, func2subr, rules
from numpy.f2py._dt_handle_clear import clear_source
from numpy.f2py._dt_handles import _handle_module
from numpy.f2py._dt_routines import prepare_routine, wrapper_handles

from . import util


def prepared(name):
    filename = ('pdt_owned_allocation.f90' if name == 'clear_unit'
                else 'pdt_routine_selections.f90')
    path = util.getpath('tests', 'src', 'derived_types',
                        filename)
    modules = crackfortran.crackfortran([path])
    module = modules[0]
    pymod = {'block': 'python module', 'name': 'routine_state', 'body': modules}
    descriptions = derived_type_rules.describe_types(pymod)
    routine = next(item for item in module['body'] if item['name'] == name)
    routine['modulename'] = module['name']
    assert prepare_routine(routine, descriptions) == []
    return routine, module


class TestPDTRoutineSelectorGeneration:
    @pytest.mark.parametrize('name', ['bump', 'sum_pair', 'bump_grid',
                                      'make_grid', 'make_cell', 'optional_read'])
    def test_handle_casts_use_canonical_native_operations(self, name):
        routine, module = prepared(name)
        uses, declarations, pre, post, body, _ = wrapper_handles(
            routine, routine['args'])
        source = '\n'.join([*uses, *declarations, *pre,
                            *body(lambda names: ['call native(' + ', '.join(
                                names.get(a, a) for a in routine['f2py_dt_args']) + ')']),
                            *post])
        assert 'f2py_pdt_native_operation' not in source
        assert 'call native(' in source
        assert routine['f2py_pdt_status'] + ' = 0' in source
        assert 'c_loc(f2py_value)' not in source
        assert 'c_f_pointer(value, f2py_value)' not in source
        assert module['f2py_pdt_bindings']

    @pytest.mark.parametrize('name', ['make_fixed_grid', 'make_fixed_cell'])
    def test_output_allocation_entities_preserve_declared_len(self, name):
        routine, _ = prepared(name)
        _, declarations, pre, post, body, _ = wrapper_handles(routine, routine['args'])
        source = '\n'.join([*declarations, *pre,
                            *body(lambda names: ['call native(' + ', '.join(
                                names.get(a, a) for a in routine['f2py_dt_args']) + ')']),
                            *post])
        assert 'n=2), allocatable :: f2py_' in source
        assert 'block' in source
        assert 'n=:), allocatable :: f2py_' not in source

    def test_scalar_bindings_dispatch_the_actual_non_target_variable(self):
        routine, module = prepared('bump')
        _, _, _, _, body, _ = wrapper_handles(routine, routine['args'])
        source = '\n'.join(body(lambda names: [f'call native({names["value"]}, amount)']))
        record = next(item for item in module['f2py_pdt_bindings']
                      if item['variable'] == 'single')
        assert f'call native({record["alias"]}, amount)' in source
        assert '=> ' + record['alias'] not in source

    def test_status_is_hidden_from_native_call_and_checked_in_c(self):
        routine, _ = prepared('bump')
        status = routine['f2py_pdt_status']
        source = func2subr.createsubrwrapper(routine)
        call_lines = [line for line in source.splitlines()
                      if 'call bump(' in line.lower()]
        assert call_lines
        assert all(status not in line for line in call_lines)
        assert all(len(line) <= 132 for line in source.splitlines())
        generated, _ = rules.buildapi(routine)
        assert status + ' != 0' in generated['body']
        assert 'PyExc_ValueError' in generated['body']

    def test_fixed_len_constraints_remain_in_the_native_wrapper(self):
        routine, _ = prepared('fixed_length')
        dt = routine['vars']['value']['f2py_dt']
        assert dt['length_parameters'] == {'n': '2'}
        _, _, _, _, body, _ = wrapper_handles(routine, routine['args'])
        source = '\n'.join(body(lambda names: [f'call native({names["value"]})']))
        assert 'f2py_pdt_length_value(value, 0,' in source
        assert 'int(2, c_int64_t)' in source

    def test_owned_clear_invalidates_before_native_payload_release(self):
        routine, _ = prepared('clear_unit')
        description = routine['vars']['value']['f2py_dt']['pdt_handle']
        source = clear_source(description)
        assert source.index('header%payload = c_null_ptr') < source.index('deallocate(old_value)')
        assert 'f2py_pdt_release_header' not in source
        assert 'status = 0' in source
        module_source = _handle_module(description)
        assert description['prefix'] + '_clear' in module_source

    def test_mutable_allocatable_dispatch_uses_owned_snapshot_or_native_entity(self):
        routine, module = prepared('clear_unit')
        _, declarations, _, _, body, _ = wrapper_handles(routine, routine['args'])
        source = '\n'.join(body(lambda names: [f'call native({names["value"]})']))
        binding = next(item for item in module['f2py_pdt_bindings']
                       if item['variable'] == 'allocated_unit')
        assert 'call native(f2py_value_allocation)' in source
        assert f'call native({binding["alias"]})' in source
        assert '_clear(value)' in source
        assert 'allocatable :: f2py_value_allocation' in source
        assert any('pointer :: f2py_value' in line for line in declarations)


@pytest.mark.slow
class TestPDTRoutineSelectors(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_routine_selections.f90')]

    def reset(self):
        module = self.module.pdt_routine_selections
        module.reset_values()
        return module

    def test_scalar_routine_updates_module_and_owned_values(self):
        module = self.reset()
        module.bump(module.single, 5)
        assert module.single.value == 16
        owned = self.module.routine_cell(n=3, value=19)
        module.bump(owned, 7)
        assert owned.value == 26
        assert owned.n == 3

    def test_two_operands_accept_mixed_native_storage(self):
        module = self.reset()
        owned = self.module.routine_cell(n=3, value=19)
        assert module.sum_pair(module.single, owned) == 30
        assert module.sum_pair(owned, module.single) == 30
        assert module.single.value == 11
        assert owned.value == 19

    def test_rank_two_selector_copyback_reaches_each_original(self):
        module = self.reset()
        values = module.fixed
        module.bump_grid(values[:, ::-1], 13)
        for index in np.ndindex((2, 3)):
            assert module.fixed[index].value == 30
            assert module.fixed[index].n == 2

    @pytest.mark.parametrize('length', [-1, 3])
    def test_vector_results_preserve_each_native_len(self, length):
        module = self.reset()
        values = module.make_grid(2, 3, length)
        assert values.shape == (2, 3)
        for index in np.ndindex(values.shape):
            assert values[index].value == 23
            assert values[index].n == length
            assert values[index].values.shape == (max(length, 0),)

    @pytest.mark.parametrize('length', [-1, 3])
    def test_output_and_function_result_preserve_native_len(self, length):
        module = self.reset()
        output = module.make_cell(length)
        result = module.cell_result(length)
        assert output.n == result.n == length
        assert output.value == 29
        assert result.value == 31
        assert output.values.shape == result.values.shape == (max(length, 0),)

    def test_fixed_len_allocatable_outputs_preserve_native_shape_and_values(self):
        module = self.reset()
        values = module.make_fixed_grid()
        assert values.shape == (2, 3)
        for index in np.ndindex(values.shape):
            assert values[index].n == 2
            assert values[index].value == 101
            np.testing.assert_array_equal(values[index].values, [103, 103])
        value = module.make_fixed_cell()
        assert value.n == 2
        assert value.value == 107
        np.testing.assert_array_equal(value.values, [109, 109])

    def test_optional_absence_and_module_selection_are_distinct(self):
        module = self.reset()
        assert module.optional_read() == -1
        assert module.optional_read(module.single) == 11

    def test_fixed_len_mismatch_rejects_before_native_mutation(self):
        module = self.reset()
        incoming = self.module.routine_cell(n=3, value=53)
        with pytest.raises(ValueError):
            module.fixed_length(incoming)
        assert incoming.n == 3
        assert incoming.value == 53

    @pytest.mark.parametrize('length', [-1, 4])
    def test_owned_allocation_replacement_refreshes_len(self, length):
        module = self.reset()
        value = self.module.routine_cell(n=2, value=59)
        module.resize_cell(value, length)
        assert value.n == length
        assert value.values.shape == (max(length, 0),)
        assert value.value == 61


@pytest.mark.slow
class TestPDTOwnedAllocatableStorage(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_owned_allocation.f90')]

    def reset(self):
        module = self.module.pdt_owned_allocation
        module.reset_values()
        return module

    def test_legal_owned_deallocation_invalidates_then_reinitializes(self):
        module = self.reset()
        value = self.module.allocation_cell(k=4, value=71)
        module.clear_unit(value)
        with pytest.raises(ReferenceError, match='unallocated'):
            _ = value.value
        with pytest.raises(ReferenceError, match='unallocated'):
            _ = value.k
        with pytest.raises(ReferenceError, match='unallocated'):
            value.value = 73
        with pytest.raises(ReferenceError, match='unallocated'):
            repr(value)
        value.__init__(k=4, value=79)
        assert value.k == 4
        assert value.value == 79
        module.clear_unit(value)
        with pytest.raises(ReferenceError, match='unallocated'):
            _ = value.value

    def test_legal_module_deallocation_returns_none_and_invalidates_old_proxy(self):
        module = self.reset()
        module.make_unit()
        value = module.allocated_unit
        assert value.value == 67
        module.clear_unit(value)
        assert module.allocated_unit is None
        with pytest.raises(ReferenceError):
            _ = value.value

    def test_owned_and_module_allocation_replacement_preserves_python_identity(self):
        module = self.reset()
        owned = self.module.allocation_cell(k=4, value=71)
        module.replace_unit(owned, 83)
        assert owned.value == 83
        assert owned.k == 4
        module.make_unit()
        held = module.allocated_unit
        module.replace_unit(held, 97)
        assert held.value == 97
        assert module.allocated_unit.value == 97
