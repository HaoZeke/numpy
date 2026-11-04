import pytest

from numpy.f2py import f90mod_rules


@pytest.mark.parametrize('typespec', ['type', 'class'])
def test_unsupported_derived_parameter_is_excluded_before_storage(monkeypatch, typespec):
    origin = dict(typespec=typespec, typename='pair',
                  attrspec=['parameter', 'optional'])
    origin['='] = 'pair(0, 0)'
    plain = dict(typespec='integer', attrspec=['parameter'])
    plain['='] = '5'
    module = dict(block='module', name='derived_constants',
                  vars=dict(origin=origin, plain=plain), body=[])
    original = f90mod_rules._parameter_variable
    selected = []

    def storage(var, block):
        assert var.get('typespec') not in ('type', 'class')
        selected.append(var)
        return original(var, block)

    monkeypatch.setattr(f90mod_rules, '_parameter_variable', storage)
    hooks, _ = f90mod_rules.buildhooks(module)
    generated = '\n'.join(hooks['f90modhooks'])
    assert '{"plain"' in generated
    assert '{"origin"' not in generated
    assert len(selected) == 1


def test_mutable_derived_variable_does_not_hide_intrinsic_neighbor():
    current = dict(typespec='type', typename='pair', attrspec=[])
    plain = dict(typespec='integer', attrspec=[])
    module = dict(block='module', name='derived_neighbors',
                  vars=dict(current=current, plain=plain), body=[])
    hooks, _ = f90mod_rules.buildhooks(module)
    generated = '\n'.join(hooks['f90modhooks'])
    assert '{"plain"' in generated
    assert '{"current"' not in generated
