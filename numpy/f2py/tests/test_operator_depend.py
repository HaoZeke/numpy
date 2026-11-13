import numpy.f2py.crackfortran as crackfortran


def test_defined_operator_name_is_not_a_regex():
    # ``operator(+)`` used to be compiled as a regex, and ``+`` made
    # ``re.error: nothing to repeat``.
    block = {
        'block': 'subroutine',
        'name': 's',
        'vars': {
            'operator(+)': {'typespec': 'integer'},
            'n': {'typespec': 'integer', '=': 'operator(+)'},
        },
        'body': [],
        'args': ['n'],
    }
    crackfortran.analyzevars(block)
    assert block['vars']['n']['typespec'] == 'integer'