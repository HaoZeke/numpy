import numpy.f2py.crackfortran as crackfortran


def test_character_hook_accepts_a_deeper_check():
    # parents[-3] is not ``vars`` when a constructor inserts an extra
    # list level. The hook used to assert and abort the module.
    item = ('attrspec', ['public'])
    parents = [
        (None, None),
        (0, []),
        ('body', []),
        (0, []),
        ('body', []),
        (0, {}),
        ('vars', {'s': {'typespec': 'character', 'charselector': {'*': '*'}}}),
        ('check', ['public']),
    ]
    out = crackfortran.character_backward_compatibility_hook(
        item, parents, None)
    assert out == item or out[0] == 'attrspec'