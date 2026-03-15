"""LEN-sized array members are accepted by the derived type check."""
from numpy.f2py.auxfuncs import is_simple_derived_type


def test_len_sized_array_is_simple():
    # real :: data(n) with integer, len :: n is a LEN array (F2018 7.5.3)
    block = {'vars': {
        'n': {'typespec': 'integer', 'attrspec': ['len']},
        'data': {'typespec': 'real', 'attrspec': [], 'dimension': ['n']},
    }}
    assert is_simple_derived_type(block)


def test_named_dimension_without_len_is_not_simple():
    block = {'vars': {
        'data': {'typespec': 'real', 'attrspec': [], 'dimension': ['n']},
    }}
    assert not is_simple_derived_type(block)
