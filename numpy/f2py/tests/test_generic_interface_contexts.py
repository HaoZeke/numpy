"""Generic specifications and ordinary identifiers retain separate contexts."""

import pytest

from numpy.f2py import crackfortran
from numpy.f2py.tests import util

_GENERIC_SPECS = ['operator(+)', 'operator(-)', 'operator(*)', 'operator(/)',
                  'operator(**)', 'operator(==)', 'operator(/=)', 'operator(<)',
                  'operator(<=)', 'operator(>)', 'operator(>=)', 'operator(.eq.)',
                  'operator(.custom.)', 'assignment(=)']


@pytest.mark.parametrize('specification', _GENERIC_SPECS)
def test_generic_specifier_is_a_name_not_a_dummy_list(specification):
    assert crackfortran._resolvenameargspattern(
        specification, interface=True) == (specification, [], None, None)


@pytest.mark.parametrize('specification,expected', [
    ('OPERATOR ( + )', 'operator(+)'),
    ('operator\t( .custom. )', 'operator(.custom.)'),
    ('ASSIGNMENT ( = )', 'assignment(=)'),
])
def test_generic_name_canonicalizes_case_and_whitespace(specification, expected):
    assert crackfortran._resolvenameargspattern(
        specification, interface=True) == (expected, [], None, None)


@pytest.mark.parametrize('name', ['operator', 'assignment'])
def test_ordinary_procedure_named_like_generic_scheme_keeps_arguments(name):
    assert crackfortran._resolvenameargspattern(
        name + '(value) result(answer)') == (name, 'value', 'answer', None)


def test_spaced_access_and_header_names_share_visibility_metadata():
    path = util.getpath('tests', 'src', 'derived_types', 'generic_spec_contexts.f90')
    module = crackfortran.crackfortran([str(path)])[0]
    names = {block['name']: block for block in module['body']
             if block.get('block') == 'interface'}
    assert set(names) == {'operator(+)', 'operator(-)', 'assignment(=)'}
    assert names['operator(+)']['args'] == []
    assert names['operator(+)']['implementedby'] == ['positive_item', 'sum_items']
    assert names['operator(-)']['implementedby'] == ['negative_item']
    assert names['assignment(=)']['implementedby'] == ['assign_integer']
    for name in names:
        assert 'public' in module['vars'][name]['attrspec']
        assert 'private' not in module['vars'][name]['attrspec']
        assert 'typespec' not in module['vars'][name]
    for name in ('operator', 'assignment'):
        assert module['vars'][name]['typespec'] == 'integer'
        assert 'public' in module['vars'][name]['attrspec']
    assert set(module['f2py_operator_procedures']) == {
        'positive_item', 'sum_items', 'negative_item', 'assign_integer'}


class TestGenericSpecificationContexts(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'generic_spec_contexts.f90')]

    def test_public_spaced_generic_reaches_private_implementations(self):
        left = self.module.scope_item(value=7)
        right = self.module.scope_item(value=5)
        assert (+left).value == 7
        assert (-left).value == -7
        assert (left + right).value == 12
        assert left.value == 7
        assert right.value == 5

    def test_spaced_assignment_generic_remains_callable(self):
        value = self.module.scope_item(value=7)
        assert value.__fortran_assign__(5) is value
        assert value.value == 55

    def test_real_data_named_operator_and_assignment_is_preserved(self):
        module = self.module.generic_spec_values
        assert module.operator == 8
        assert module.assignment == 9

    def test_ordinary_functions_named_operator_and_assignment_are_callable(self):
        module = self.module.ordinary_generic_words
        assert module.operator(4) == 7
        assert module.assignment(4) == 9
