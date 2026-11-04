"""Generic accessibility and procedure lists retain operator semantics."""

import pytest

from numpy.f2py import crackfortran

from . import util

_ACCESS = [(0, 'operator(+)', 'public'), (0, 'operator(-)', 'public'),
           (0, 'operator(==)', 'public'), (0, 'operator(*)', 'private'),
           (0, 'assignment(=)', 'public'), (1, 'operator(-)', 'private')]
_PROCEDURES = [
    ('operator(+)', ['small_add', 'small_positive', 'small_add_int',
                     'int_add_small', 'small_add_long', 'small_add_real',
                     'small_add_double', 'small_add_wide', 'tracked_add',
                     'small_array_sum']),
    ('operator(-)', ['small_negative', 'small_sub_int', 'int_sub_small']),
    ('assignment(=)', ['small_assign_int', 'small_assign_small',
                       'tracked_assign_small', 'wide_assign_int',
                       'pair_assign_int']),
]


@pytest.mark.parametrize('module_index,name,access', _ACCESS)
def test_generic_visibility_metadata_is_not_implicitly_typed(
        module_index, name, access):
    source = util.getpath('tests', 'src', 'derived_types', 'private_operators.f90')
    module = crackfortran.crackfortran([str(source)])[module_index]
    metadata = module['vars'][name]
    assert access in metadata['attrspec']
    assert 'typespec' not in metadata
    assert 'operator' not in module['vars']
    assert 'assignment' not in module['vars']


@pytest.mark.parametrize('name,procedures', _PROCEDURES)
def test_module_procedure_statements_preserve_all_overloads(name, procedures):
    source = util.getpath('tests', 'src', 'derived_types', 'operator_dispatch.f90')
    module, = crackfortran.crackfortran([str(source)])
    interface, = [block for block in module['body'] if block.get('name') == name]
    assert interface['implementedby'] == procedures


def test_named_generic_and_double_colon_procedure_lists():
    source = util.getpath('tests', 'src', 'derived_types',
                          'multi_statement_operators.f90')
    module, = crackfortran.crackfortran([str(source)])
    assert 'public' in module['vars']['add']['attrspec']
    assert 'typespec' not in module['vars']['add']
    interface, = [block for block in module['body']
                  if block.get('name') == 'operator(+)']
    assert interface['implementedby'] == ['positive_value', 'add_values']


class TestMultiStatementOperators(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                           'multi_statement_operators.f90')]

    def test_unary_plus_from_first_statement(self):
        value = self.module.list_value(value=7)
        result = +value
        assert type(result) is type(value)
        assert result.value == 7
        assert value.value == 7

    def test_binary_plus_from_second_statement(self):
        left = self.module.list_value(value=7)
        right = self.module.list_value(value=5)
        result = left + right
        assert type(result) is type(left)
        assert result.value == 12
        assert left.value == 7
        assert right.value == 5
