"""Explicit callback interfaces register complete native callback definitions."""

import pytest

from numpy.f2py import crackfortran

from . import util

_CASES = [
    ('saved_interface_top_level.f90', None, 'saved_top_function', 'function'),
    ('saved_interface_top_level.f90', None, 'saved_top_subroutine', 'function'),
    ('saved_interface_top_level.f90', 'saved_signature_module',
     'saved_child_function', 'function'),
    ('explicit_subroutine_callback.f90', None,
     'explicit_subroutine_callback', 'subroutine'),
]


@pytest.mark.parametrize('filename,module_name,routine_name,callback_kind', _CASES)
def test_explicit_callback_has_registered_definition(
        filename, module_name, routine_name, callback_kind):
    source = util.getpath('tests', 'src', 'callback', filename)
    blocks = crackfortran.crackfortran([str(source)])
    routines = blocks
    if module_name is not None:
        module, = [block for block in blocks if block.get('name') == module_name]
        routines = module['body']
    routine, = [block for block in routines if block.get('name') == routine_name]

    assert routine['externals'] == ['callback']
    assert 'callback' not in routine['interfaced']
    user_module, = [block for block in blocks
                   if block['block'] == 'python module'
                   and block['name'] in routine['use']]
    callback, = [definition for interface in user_module['body']
                 for definition in interface['body']
                 if definition.get('name') == 'callback']
    assert callback['block'] == callback_kind
    assert callback['vars']['value']['kindselector']['kind'] == '8'
    if callback_kind == 'function':
        result = callback['result']
    else:
        result = 'output'
    assert callback['vars'][result]['kindselector']['kind'] == '8'
    saved = [line.strip() for line in routine['saved_interface'].splitlines()]
    assert 'external callback' not in saved
    rendered = [line.strip() for line in
                crackfortran.crack2fortran([routine]).splitlines()]
    assert rendered.count('external callback') == 1


class TestExplicitSubroutineCallback(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'callback',
                           'explicit_subroutine_callback.f90')]

    def test_integer8_subroutine_callback(self):
        value = 2**40 + 3
        increment = 2**35 + 17
        observed = []

        def callback(argument):
            observed.append(argument)
            return argument + increment

        answer = self.module.explicit_subroutine_callback(callback, value)
        assert answer == value + increment
        assert observed == [value]
