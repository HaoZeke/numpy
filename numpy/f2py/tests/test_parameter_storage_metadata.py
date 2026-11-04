import pytest

import numpy as np
from numpy.f2py import crackfortran, f90mod_rules
from numpy.f2py._parameter_metadata import storage_parameters


def character(length, dimensions=None):
    var = dict(typespec='character', attrspec=['parameter'],
               charselector=dict(len=str(length)))
    var['='] = "'a'"
    if dimensions is not None:
        var['dimension'] = list(dimensions)
    return var


def integer(value, kind=None):
    var = dict(typespec='integer', attrspec=['parameter'])
    var['='] = value
    if kind is not None:
        var['kindselector'] = dict(kind=str(kind))
    return var


def module(variables):
    return dict(block='module', name='storage_metadata', vars=variables, body=[])


def test_descriptor_rejects_length_without_materializing_target(monkeypatch):
    var = character(int(np.iinfo(np.intc).max) + 1, ['0'])

    def forbidden(*args, **kwargs):
        raise AssertionError('Descriptor validation needs no parameter values')

    monkeypatch.setattr(crackfortran, '_get_typed_parameters', forbidden)
    with pytest.raises(ValueError, match='extent exceeds'):
        f90mod_rules._parameter_variable(var, module(dict(word=var)))


def test_only_integer_selector_dependencies_are_evaluated(monkeypatch):
    target = character('n', ['0'])
    variables = dict(n=integer('5 / 2 * 2', 8), word=target,
                     unrelated=character(int(np.iinfo(np.intc).max) + 1))
    calls = []
    original = crackfortran._get_typed_parameters

    def evaluate(selected, *args, **kwargs):
        calls.append(set(selected))
        return original(selected, *args, **kwargs)

    monkeypatch.setattr(crackfortran, '_get_typed_parameters', evaluate)
    result = f90mod_rules._parameter_variable(target, module(variables))
    assert result['charselector']['f2py_len'] == '4'
    assert result['dimension'] == ['0']
    assert calls == [{'n'}]


def test_len_and_kind_inquiries_do_not_materialize_character(monkeypatch):
    variables = dict(word=character(2**30), n=integer('len(word, kind=8)', 8))
    params = storage_parameters(variables, {})
    original = crackfortran._get_typed_parameters
    calls = []

    def evaluate(selected, *args, **kwargs):
        calls.extend(selected)
        assert 'word' not in selected
        return original(selected, *args, **kwargs)

    monkeypatch.setattr(crackfortran, '_get_typed_parameters', evaluate)
    assert crackfortran._eval_scalar('n', params) == str(2**30)
    assert crackfortran._eval_scalar('kind(word)', params) == '1'
    assert params['n'].kind == 8
    assert calls == ['n']


def test_size_and_array_element_kind_use_declarations(monkeypatch):
    values = integer('(/1_8, 2_8/)', 8)
    values['attrspec'].append('dimension(2:3)')
    params = storage_parameters(dict(values=values), {})

    def forbidden(*args, **kwargs):
        raise AssertionError('Array inquiry needs no array contents')

    monkeypatch.setattr(crackfortran, '_get_typed_parameters', forbidden)
    assert crackfortran._eval_scalar('size(values, kind=8)', params) == '2'
    assert crackfortran._eval_scalar('kind(values(2))', params) == '8'


def test_model_inquiry_preserves_declared_integer_kind(monkeypatch):
    params = storage_parameters(dict(value=integer('7_8', 8)), {})

    def forbidden(*args, **kwargs):
        raise AssertionError('Type inquiry needs no parameter initializer')

    monkeypatch.setattr(crackfortran, '_get_typed_parameters', forbidden)
    result = crackfortran._fortran_eval('huge(value)', {}, params)
    assert result == 2**63 - 1
    assert result.kind == 8


def test_needed_character_contents_retain_typed_value_semantics():
    params = storage_parameters(dict(letter=character(1)), {})
    assert crackfortran._eval_scalar('iachar(letter)', params) == str(ord('a'))


def test_unrelated_parameter_cycle_is_not_evaluated():
    params = storage_parameters(dict(n=integer('3'), bad=integer('bad + 1')), {})
    assert crackfortran._eval_scalar('n', params) == '3'
    assert 'bad' not in dict(params)


def test_storage_queries_keep_use_parameter_kinds():
    inherited = dict(k=crackfortran._Integer(8, 4),
                     count=crackfortran._Integer(5, 8))
    params = storage_parameters(dict(n=integer('count / 2 * 2', 'k')), inherited)
    assert crackfortran._eval_scalar('n', params) == '4'
    assert params['n'].kind == 8


def test_size_inquiry_can_infer_implied_parameter_shape():
    values = integer('(/2, 3, 4/)')
    values['attrspec'].append('dimension(*)')
    params = storage_parameters(dict(values=values), {})
    assert crackfortran._eval_scalar('size(values)', params) == '3'


def test_len_inquiry_handles_assumed_length_parameter_array():
    values = character('*')
    values['attrspec'].append('dimension(2)')
    values['='] = "(/'ab', 'cd'/)"
    params = storage_parameters(dict(values=values), {})
    assert crackfortran._eval_scalar('len(values)', params) == '2'


def test_kind_inquiry_can_use_initializer_when_declaration_has_no_type():
    params = storage_parameters(dict(value=dict(attrspec=['parameter'],
                                               **{'=': '7_8'})), {})
    assert crackfortran._eval_scalar('kind(value)', params) == '8'


def test_local_parameter_overrides_inherited_case_alias():
    params = storage_parameters(dict(N=integer('3', 8)),
                                dict(n=crackfortran._Integer(9, 4)))
    assert crackfortran._eval_scalar('n', params) == '3'
    assert params['n'].kind == 8


def test_size_inquiry_rejects_noninteger_declared_extent():
    values = integer('(/2, 3/)')
    values['dimension'] = ['2.5']
    params = storage_parameters(dict(values=values), {})
    with pytest.raises(TypeError, match='extent is an integer'):
        crackfortran._fortran_eval('size(values)', {}, params)


@pytest.mark.parametrize('expression, local, expected', [
    ('kind(value)', {'value': crackfortran._Integer(0, 4)}, 4),
    ('len(word)', {'word': crackfortran._Character('xy', 1)}, 2),
])
def test_inquiries_respect_concrete_local_bindings(expression, local, expected):
    params = storage_parameters(dict(value=integer('7_8', 8),
                                     word=character(2**30)), {})
    result = crackfortran._fortran_eval(expression, params, local)
    assert result == expected


def test_locally_bound_array_can_shadow_an_intrinsic_name():
    params = storage_parameters(dict(value=integer('1_8', 8)), {})
    local = dict(kind=crackfortran._ParameterArray(
                     {1: crackfortran._Integer(77, 4)}),
                 value=crackfortran._Integer(1, 8))
    assert crackfortran._fortran_eval('kind(value)', params, local) == 77


@pytest.mark.parametrize('module_index', [None, 1])
def test_size_dim_uses_callers_local_index(module_index):
    values = integer('0')
    values['dimension'] = ['2', '3']
    variables = dict(values=values)
    if module_index is not None:
        variables['i'] = integer(str(module_index))
    params = storage_parameters(variables, {})
    result = crackfortran._fortran_eval(
        'size(values, dim=i)', params, {'i': crackfortran._Integer(2, 4)})
    assert result == 3
    assert result.kind == 4


@pytest.mark.parametrize('module_kind', [None, 4])
def test_len_result_kind_uses_callers_local_kind(module_kind):
    variables = dict(word=character(5))
    if module_kind is not None:
        variables['k'] = integer(str(module_kind))
    params = storage_parameters(variables, {})
    result = crackfortran._fortran_eval(
        'len(word, kind=k)', params, {'k': crackfortran._Integer(8, 4)})
    assert result == 5
    assert result.kind == 8


def test_size_optional_arguments_keep_translated_literal_values():
    values = integer('0')
    values['dimension'] = ['2', '3']
    params = storage_parameters(dict(values=values), {})
    result = crackfortran._fortran_eval('size(values, dim=2, kind=8)', {}, params)
    assert result == 3
    assert result.kind == 8


@pytest.mark.parametrize('expression', ['len(word)', 'size(values)'])
def test_declaration_metadata_retains_its_module_scope(expression):
    values = integer('0')
    values['dimension'] = ['n']
    params = storage_parameters(dict(n=integer('5'), word=character('n'),
                                     values=values), {})
    result = crackfortran._fortran_eval(
        expression, params, {'n': crackfortran._Integer(9, 4)})
    assert result == 5


def test_optional_argument_lookup_keeps_unresolved_string_fallback():
    values = integer('0')
    values['dimension'] = ['2', '3']
    params = storage_parameters(dict(i=integer('2'), values=values), {})
    result = crackfortran._fortran_eval(
        'size(values, dim=i)', params, {'i': 'unresolved_source_text'})
    assert result == 3
