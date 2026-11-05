"""Native array-copy interface identities and descriptor requirements."""

import pytest

from numpy.f2py._dt_numeric_handles import _numeric_fortran, _numeric_root


def test_component_symbols_bind_owner_and_specialization():
    first = _numeric_root('owner', 'cell_k4', 'plane')
    assert first == _numeric_root('OWNER', 'CELL_K4', 'PLANE')
    assert len({first, _numeric_root('other', 'cell_k4', 'plane'),
                _numeric_root('owner', 'cell_k8', 'plane')}) == 3


def test_maximum_native_identifier_lengths_produce_short_symbols():
    root = _numeric_root('m' * 63, 't' * 63, 'v' * 63)
    assert len(root + '_shape') <= 63
    assert len(root + '_copy') <= 63
    assert root.isidentifier()


@pytest.mark.parametrize('typespec,kind', [('real', '8'), ('integer', '4')])
def test_len_array_interface_copies_into_owned_buffer(typespec, kind):
    member = {'typespec': typespec, 'kindselector': {'kind': kind},
              'dimension': ['n', '3']}
    block = {'block': 'type', 'name': 'cell',
             'parent_block': {'block': 'module', 'name': 'owner'},
             'vars': {'n': {'typespec': 'integer', 'attrspec': ['len'], '=': '2'},
                      'plane': member}}
    lines, removed = _numeric_fortran('cell', 'cell(n=n)', block, {'plane': member})
    source = '\n'.join(lines)
    assert 'c_loc(' not in source.lower()
    assert 'call c_f_pointer(buffer, values, dims)' in source
    assert 'values = object%plane' in source
    assert 'shape(object%plane, kind=c_size_t)' in source
    assert 'any(shape(object%plane, kind=c_size_t) /= dims)' in source
    assert removed == {'f2py_get_cell_plane', 'f2py_get_cell_plane_data'}
