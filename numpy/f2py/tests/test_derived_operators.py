"""Native operator dispatch, scalar conversion and result lifetime tests."""

import gc
import re

import pytest

import numpy as np
from numpy.f2py import crackfortran, derived_type_rules

from . import util


class TestDerivedOperatorGeneration:
    def test_unique_slots_and_native_result_deallocators(self):
        path = util.getpath('tests', 'src', 'derived_types',
                            'operator_dispatch.f90')
        module = crackfortran.crackfortran([str(path)])[0]
        hooks = derived_type_rules.buildhooks(module)
        source = '\n'.join(hooks['f90modhooks'])
        names = re.findall(r'\n(Py\w+_nb_\w+)\(', source)
        assert len(names) == len(set(names))
        assert 'Pysmall_value_nb_positive' in names
        assert 'Pysmall_value_nb_negative' in names
        assert 'Pysmall_value_nb_add' in names
        assert 'Pysmall_value_nb_subtract' in names
        assert 'f2py_op_small_value_small_add_free(value)' in source
        assert 'f2py_op_tracked_value_tracked_add_free(value)' in source
        assert ('if (result == NULL) '
                'f2py_op_small_value_small_add_free(value);') in source
        assert ('if (result == NULL) '
                'f2py_op_tracked_value_tracked_add_free(value);') in source
        assert 'f2py_op_wide_value_adopt' in source
        types = [b for b in module['body'] if b.get('block') == 'type']
        wrappers = derived_type_rules.generate_fortran_wrappers(
            'operator_dispatch_mod', types, module_block=module)
        assert wrappers.count('function f2py_op_small_value_small_add(') == 1
        assert ('subroutine f2py_op_small_value_small_add_free(f2py_op_handle)'
                in wrappers)
        assert 'deallocate(f2py_op_value)' in wrappers
        assert ('source=small_add_int(f2py_op_operand0, f2py_op_arg1)' in wrappers)
        assert 'integer(c_int), value :: f2py_op_arg1' in wrappers
        assert 'f2py_op_small_value_small_array_sum' not in wrappers
        assert 'f2py_op_small_value_assignment' in source
        assert '"__fortran_assign__"' in source
        assert '"assign_from"' in source
        assert 'source=f2py_op_operand1' in wrappers


@pytest.mark.slow
class TestDerivedOperatorDispatch(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'operator_dispatch.f90')]

    def test_multiple_binary_and_unary_signatures(self):
        left = self.module.small_value(value=7)
        right = self.module.small_value(value=5)
        result = left + right
        assert type(result) is type(left)
        assert result.value == 12
        assert (+left).value == 7
        assert (-left).value == -7
        assert left.value == 7
        assert right.value == 5

    def test_integer_and_reflected_arithmetic(self):
        value = self.module.small_value(value=10)
        assert (value + 3).value == 13
        assert (3 + value).value == 13
        assert (value - 3).value == 7
        assert (3 - value).value == -7

    def test_scalar_kind_selection(self):
        value = self.module.small_value(value=3)
        assert (value + 2).value == 5
        assert (value + np.int32(2)).value == 5
        assert (value + np.int64(2)).value == 1005
        assert (value + 2.0).value == 5
        assert (value + np.float32(2)).value == 5
        assert (value + np.float64(2)).value == 105

    def test_integer_conversion_overflow(self):
        value = self.module.small_value(value=3)
        with pytest.raises(OverflowError, match='operator integer out of range'):
            value + 2**40
        with pytest.raises(OverflowError):
            value + 2**80

    def test_index_conversion_can_reinitialize_operand(self):
        value = self.module.small_value(value=3)

        class Reinitialize:
            def __index__(self):
                value.__init__(value=20)
                return 4

        result = value + Reinitialize()
        assert result.value == 24
        assert value.value == 20

    def test_index_error_propagates(self):
        value = self.module.small_value(value=3)

        class Failure:
            def __index__(self):
                raise ValueError('index conversion failed')

        with pytest.raises(ValueError, match='index conversion failed'):
            value + Failure()
        assert value.value == 3

    def test_mixed_derived_operands_and_result_type(self):
        left = self.module.small_value(value=3)
        right = self.module.wide_value(value=8)
        result = left + right
        assert type(result) is type(right)
        assert result.value == 11
        assert left.value == 3
        assert right.value == 8

    def test_scalar_operator_result(self):
        left = self.module.small_value(value=3)
        right = self.module.small_value(value=8)
        assert left * right == 24
        assert isinstance(left * right, int)

    def test_complex_operand_and_scalar_result(self):
        value = self.module.small_value(value=3)
        assert value * (2 + 4j) == 6 + 12j
        assert value * np.complex128(2 + 4j) == 6 + 12j

    def test_comparison_overloads_and_reflection(self):
        value = self.module.small_value(value=7)
        assert value == self.module.small_value(value=7)
        assert value != self.module.small_value(value=8)
        assert 7 == value
        assert value == 7
        assert 6 != value
        assert 6 < value
        assert value > 6
        assert not (8 < value)
        assert not (value > 8)

    def test_unsupported_operands_return_not_implemented(self):
        value = self.module.small_value(value=7)
        assert value.__add__('text') is NotImplemented
        assert value.__eq__('text') is NotImplemented
        with pytest.raises(TypeError):
            value + 'text'

    def test_logical_operand_conversion(self):
        value = self.module.small_value(value=7)
        truth = True
        falsehood = False
        assert truth == value
        assert np.bool_(True) == value
        assert falsehood != value

    def test_uninitialized_operand_raises(self):
        cls = self.module.small_value
        initialized = cls(value=7)
        uninitialized = cls.__new__(cls)
        with pytest.raises(RuntimeError, match='operand not initialized'):
            initialized + uninitialized

    def test_native_allocated_bindc_results_remain_independent(self):
        value = self.module.small_value(value=1)
        for _ in range(128):
            result = value + value + value
            assert result.value == 3
            result.value = 9
            assert value.value == 1
            del result
        gc.collect()
        assert value.value == 1

    def test_operator_result_calls_fortran_finalizer(self):
        left = self.module.tracked_value(value=3)
        right = self.module.tracked_value(value=8)
        result = left + right
        assert result.value == 11
        baseline = self.module.operator_dispatch_mod.get_finalized()
        del result
        gc.collect()
        assert self.module.operator_dispatch_mod.get_finalized() == baseline + 1
        assert left.value == 3
        assert right.value == 8

    def test_retained_result_owns_its_fortran_allocation(self):
        left = self.module.tracked_value(value=4)
        right = self.module.tracked_value(value=5)
        result = left + right
        del left, right
        gc.collect()
        baseline = self.module.operator_dispatch_mod.get_finalized()
        assert result.value == 9
        del result
        gc.collect()
        assert self.module.operator_dispatch_mod.get_finalized() == baseline + 1

    def test_array_operator_procedure_keeps_ordinary_interface(self):
        values = [self.module.small_value(value=n) for n in (1, 2, 3, 4)]
        assert self.module.operator_dispatch_mod.small_array_sum(
            values[:2], values[2:]) == 10

    def test_defined_assignment_from_integer(self):
        value = self.module.small_value(value=1)
        result = value.assign_from(7)
        assert result is value
        assert value.value == 17
        assert value.__fortran_assign__(3) is value
        assert value.value == 13

    def test_defined_assignment_from_record(self):
        destination = self.module.small_value(value=1)
        source = self.module.small_value(value=7)
        assert destination.assign_from(source) is destination
        assert destination.value == 57
        assert source.value == 7

    def test_defined_self_assignment_materializes_rhs(self):
        value = self.module.small_value(value=7)
        assert value.assign_from(value) is value
        assert value.value == 57

    def test_defined_assignment_rejects_unsupported_source(self):
        value = self.module.small_value(value=7)
        with pytest.raises(TypeError, match='no matching Fortran assignment interface'):
            value.assign_from('unsupported')
        assert value.value == 7

    def test_defined_assignment_mutates_opaque_destination(self):
        destination = self.module.tracked_value(value=1)
        source = self.module.small_value(value=7)
        before = self.module.operator_dispatch_mod.get_finalized()
        assert destination.assign_from(source) is destination
        assert destination.value == 14
        assert source.value == 7
        assert self.module.operator_dispatch_mod.get_finalized() == before + 1

    def test_assignment_alias_preserves_type_bound_method(self):
        value = self.module.wide_value(value=7)
        assert value.assign_from(2) == 10009
        assert value.value == 7
        assert value.__fortran_assign__(3) is value
        assert value.value == 30

    def test_assignment_preserves_bindc_component_view(self):
        value = self.module.fixed_pair()
        view = value.value
        assert value.assign_from(8) is value
        assert view.tolist() == [8, 9]
        assert value.value.tolist() == [8, 9]
