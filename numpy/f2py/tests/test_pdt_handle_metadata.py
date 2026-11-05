"""Source-generation contracts for parameterized native handle metadata."""

from copy import deepcopy

import pytest

from numpy.f2py._dt_handles import (
    _fortran_source,
    _handle_description,
    _handle_module,
    _handle_operation_parts,
    _replace_fortran_name,
)


def _block(owner='records', name='cell'):
    return {
        'block': 'type', 'name': name,
        'parent_block': {'block': 'module', 'name': owner},
        'vars': {
            'k': {'typespec': 'integer', 'attrspec': ['kind'], '=': '4'},
            'n': {'typespec': 'integer', 'attrspec': ['len']},
        },
    }


def _binding(identifier, storage, module, *, target=False, readonly=False):
    return {
        'id': identifier, 'storage_id': storage,
        'module': module, 'variable': 'single', 'alias': 'f2py_' + module,
        'rank': 0, 'pointer': False, 'target': target, 'readonly': readonly,
    }


def test_ordinary_types_have_no_descriptor_box():
    block = _block()
    block['vars'] = {'value': {'typespec': 'real'}}
    assert _handle_description(block) is None


def test_owner_and_kind_determine_identity_without_len_value():
    block = _block()
    four = _handle_description(block, 'cell(k=4,n=2)')
    other_length = _handle_description(block, 'cell(k=4,n=99)')
    eight = _handle_description(block, 'cell(k=8,n=2)')
    other_owner = _handle_description(_block('other'), 'cell(k=4,n=2)')
    assert four['identity_number'] == other_length['identity_number']
    assert len({four['identity_number'], eight['identity_number'],
                other_owner['identity_number']}) == 3
    assert 'k=4' in four['pointer_spec']
    assert 'n=:' in four['pointer_spec']
    assert 'n=*' in four['dummy_spec']


def test_import_alias_keeps_the_declaring_identity():
    native = _block()
    imported = deepcopy(native)
    imported['name'] = 'local_alias'
    imported['parent_block']['name'] = 'consumer'
    imported['f2py_origin_name'] = 'cell'
    imported['f2py_origin_module'] = 'records'
    assert (_handle_description(native)['identity_number']
            == _handle_description(imported)['identity_number'])


def test_required_kind_is_not_substituted_by_a_default():
    block = _block()
    del block['vars']['k']['=']
    with pytest.raises(ValueError, match='concrete KIND'):
        _handle_description(block)
    assert _handle_description(block, kind_override={'k': 8})['kinds'] == {'k': 8}


def test_reexports_share_one_native_dispatch_case():
    block = _block()
    identity = _handle_description(block)['identity_number']
    first = _binding(1, 7, 'records')
    alias = _binding(2, 7, 'facade')
    block['f2py_pdt_bindings'] = [
        dict(first, identity_number=identity), dict(alias, identity_number=identity)]
    description = _handle_description(block)
    assert len(description['bindings']) == 1
    assert description['bindings'][0]['id'] == 7
    assert [item['id'] for item in block['f2py_pdt_bindings']] == [1, 2]


def test_readonly_selector_is_absent_from_native_write_branches():
    description = _handle_description(_block())
    description['bindings'] = [
        _binding(7, 7, 'records', readonly=True),
        _binding(9, 9, 'writable'),
    ]
    uses, declarations, body = _handle_operation_parts(
        [], [], ['object%value = 17'],
        {'object': {'handle': 'handle', 'description': description}},
        write_targets={'object'})
    generated = '\n'.join(body)
    assert 'case (7)' not in generated
    assert 'case (9)' in generated
    assert 'f2py_writable%value = 17' in generated
    assert 'f2py_records%value = 17' not in generated
    assert any('type(f2py_pdt_handle)' in line for line in declarations)
    assert any('use writable' in line for line in uses)


def test_selector_substitution_preserves_component_names_and_literals():
    source = ['object%object = 1', 'print *, "object", object%value ! object']
    assert _replace_fortran_name(source, 'object', 'module_value') == [
        'module_value%object = 1',
        'print *, "object", module_value%value ! object',
    ]


def test_fixed_box_uses_complete_deferred_len_pointer_interfaces():
    description = _handle_description(_block())
    source = _fortran_source(_handle_module(description))
    assert f'type({description["pointer_spec"]}), pointer, intent(inout) :: value' in source
    assert f'c_loc({description["box"]})' not in source
    assert 'call c_f_pointer(handle, value)' not in source
    assert 'c_loc(box)' in source
    assert all(len(line) <= 132 for line in source.splitlines())


def test_long_owner_and_type_names_keep_the_free_form_source_limit():
    description = _handle_description(_block('m' * 63, 't' * 63))
    source = _fortran_source(_handle_module(description))
    assert all(len(line) <= 132 for line in source.splitlines())
    joined = source.replace('&\n', '').replace('  & ', '')
    assert 'm' * 63 in joined
    assert 't' * 63 in joined
