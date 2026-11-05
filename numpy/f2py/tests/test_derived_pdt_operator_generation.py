"""Canonical operator descriptions agree with native handle dispatch."""

from numpy.f2py import crackfortran, derived_type_rules
from numpy.f2py._dt_helpers import _module_types
from numpy.f2py._dt_operator_handles import _operator_handle_signature
from numpy.f2py.tests import util


def operator_module(name):
    path = util.getpath('tests', 'src', 'derived_types',
                        'pdt_operator_selectors.f90')
    return next(module for module in crackfortran.crackfortran([str(path)])
                if module['name'] == name)


def operator_procedure(module, name):
    procedures = [*module.get('body', []),
                  *module.get('f2py_operator_procedures', [])]
    return next(proc for proc in procedures if proc['name'] == name)


class TestDerivedPDTOperatorGeneration:
    def test_private_specific_uses_canonical_operand_and_result_identity(self):
        module = operator_module('pdt_operator_values')
        types, _, _ = _module_types(module)
        proc = operator_procedure(module, 'add_cells')
        signature = _operator_handle_signature(proc, types)
        assert signature['active']
        identities = [entry['description']['identity']
                      for entry in [*signature['args'], signature['result']]]
        assert identities == ['pdt_operator_values:op_cell:k=4'] * 3
        capsule = 'f2py.' + types['op_cell'].get('f2py_c_name', 'op_cell') + '_k4'
        assert all(entry['capsule'] == capsule
                   for entry in [*signature['args'], signature['result']])

    def test_mixed_kind_native_result_has_distinct_canonical_identity(self):
        module = operator_module('pdt_operator_values')
        types, _, _ = _module_types(module)
        proc = operator_procedure(module, 'mixed_cells')
        signature = _operator_handle_signature(proc, types)
        left, right = signature['args']
        assert left['description']['kinds'] == {'k': 4}
        assert right['description']['kinds'] == {'k': 8}
        assert (signature['result']['description']['identity']
                == right['description']['identity'])
        assert left['capsule'] != right['capsule']

    def test_assumed_and_fixed_lengths_remain_distinct(self):
        module = operator_module('pdt_operator_lengths')
        types, _, _ = _module_types(module)
        negate = _operator_handle_signature(
            operator_procedure(module, 'negate_cell'), types)
        fixed = _operator_handle_signature(
            operator_procedure(module, 'add_fixed'), types)
        assert negate['args'][0]['lengths'] == []
        assert 'n=:' in negate['args'][0]['description']['pointer_spec']
        assert negate['result']['lengths'] == []
        assert fixed['args'][0]['lengths'] == [
            dict(name='n', expression='2', constant=2)]

    def test_native_dispatch_owns_results_and_exports_status(self):
        module = operator_module('pdt_operator_lengths')
        types = [block for block in module['body'] if block.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            module['name'], types, module_block=module)
        procedure = source.split('function f2py_op_len_cell_negate_cell(', 1)[1]
        procedure = procedure.split('end function f2py_op_len_cell_negate_cell', 1)[0]
        assert 'f2py_op_status' in procedure
        assert 'integer(c_int), intent(out) :: f2py_op_status' in procedure
        assert 'f2py_pdt_validate(' in procedure
        assert '_unwrap(' in procedure
        assert '_adopt(f2py_op_value)' in procedure
        assert 'c_loc(f2py_op_value)' not in procedure
        assert 'call c_f_pointer(f2py_op_arg0' not in procedure
        assert 'source=negate_cell(' in procedure
        assert 'f2py_op_status = 0_c_int' in procedure

    def test_pointer_dummy_metadata_is_shared_with_native_dispatch(self):
        module = operator_module('pdt_operator_values')
        types, _, _ = _module_types(module)
        signature = _operator_handle_signature(
            operator_procedure(module, 'pointer_quotient'), types)
        assert signature['active']
        assert all(entry['pointer'] for entry in signature['args'])
        assert all(entry['intent_in'] for entry in signature['args'])

    def test_pointer_and_allocatable_len_modes_are_explicit(self):
        module = operator_module('pdt_operator_lengths')
        types, _, _ = _module_types(module)
        assumed = _operator_handle_signature(
            operator_procedure(module, 'assumed_pointer_total'), types)['args'][0]
        deferred = _operator_handle_signature(
            operator_procedure(module, 'deferred_pointer_total'), types)['args'][0]
        fixed = _operator_handle_signature(
            operator_procedure(module, 'allocated_fixed_equal'), types)['args'][0]
        assert assumed['length_parameters'] == {'n': '*'}
        assert assumed['deferred_lengths'] == []
        assert deferred['length_parameters'] == {'n': ':'}
        assert deferred['deferred_lengths'] == ['n']
        assert fixed['length_parameters'] == {'n': '2'}
        assert fixed['deferred_lengths'] == []
        assert fixed['allocatable']

    def test_ordinary_operator_keeps_its_abi(self):
        path = util.getpath('tests', 'src', 'derived_types', 'operator_dispatch.f90')
        module = crackfortran.crackfortran([str(path)])[0]
        types, _, _ = _module_types(module)
        signature = _operator_handle_signature(
            operator_procedure(module, 'small_add'), types)
        assert signature == dict(active=False, args=[None, None], result=None)
        blocks = [block for block in module['body'] if block.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            module['name'], blocks, module_block=module)
        procedure = source.split('function f2py_op_small_value_small_add(', 1)[1]
        procedure = procedure.split('end function f2py_op_small_value_small_add', 1)[0]
        assert 'f2py_op_status' in procedure
        declaration = procedure.split('bind(c)', 1)[0]
        assert 'f2py_op_arg0, f2py_op_arg1)' in declaration
        assert 'f2py_op_arg1, f2py_op_status)' not in declaration
