"""Declared LEN modes select only compatible native allocation entities."""

import pytest

from numpy.f2py._dt_handle_modes import binding_compatible


@pytest.mark.parametrize('allocatable,pointer,rank,actual_deferred,dummy_deferred,actual,expected,allowed', [
    (True, False, 0, ['n'], ['n'], ':', ':', True),
    (True, False, 0, [], [], '2', '2', True),
    (True, False, 0, [], ['n'], '2', ':', False),
    (True, False, 1, ['n'], ['n'], ':', ':', False),
    (False, False, 0, [], [], '2', '2', False),
    (True, False, 0, [], [], '3', '2', False),
    (False, True, 0, [], ['n'], '2', ':', False),
    (False, True, 0, ['n'], ['n'], ':', ':', True),
    (False, True, 0, [], [], '2', '*', True),
])
def test_native_entity_len_modes(allocatable, pointer, rank, actual_deferred,
                                dummy_deferred, actual, expected, allowed):
    binding = {'allocatable': allocatable, 'pointer': pointer, 'rank': rank,
               'deferred_lengths': actual_deferred,
               'length_parameters': {'n': actual}}
    targets = {'value'} if not pointer else set()
    assert binding_compatible(binding, 'value', targets,
                               {'value': dummy_deferred},
                               {'value': {'n': expected}}) is allowed
