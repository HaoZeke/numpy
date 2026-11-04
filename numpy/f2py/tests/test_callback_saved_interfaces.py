"""Complete callback signatures for external and module routines."""

import pytest

from numpy.f2py import crackfortran

from . import util

_SOURCE = util.getpath('tests', 'src', 'callback', 'saved_interface_top_level.f90')
_ROUTINES = [(None, 'saved_top_function'),
             (None, 'saved_top_subroutine'),
             ('saved_signature_module', 'saved_child_function')]
_INTEGER8_VALUE = 2**40 + 3
_INTEGER8_INCREMENT = 2**35 + 17


@pytest.mark.parametrize('module_name,routine_name', _ROUTINES)
def test_saved_callback_signature_has_full_declarations(module_name, routine_name):
    blocks = crackfortran.crackfortran([str(_SOURCE)])
    if module_name is not None:
        module, = [block for block in blocks if block.get('name') == module_name]
        blocks = module['body']
    routine, = [block for block in blocks if block.get('name') == routine_name]
    lines = [line.strip() for line in routine['saved_interface'].splitlines()]
    assert 'function callback(value) result (callback_result)' in lines
    assert 'integer(kind=8) :: callback_result' in lines
    assert 'external callback' not in lines
    assert 'external callback' in crackfortran.crack2fortran([routine])


@pytest.mark.slow
class TestSavedCallbackSignatures(util.F2PyTest):
    sources = [_SOURCE]

    @pytest.mark.parametrize('module_name,routine_name', _ROUTINES)
    def test_native_integer8_callback(self, module_name, routine_name):
        module = (self.module if module_name is None
                  else getattr(self.module, module_name))
        routine = getattr(module, routine_name)
        observed = []

        def callback(value):
            observed.append(value)
            return value + _INTEGER8_INCREMENT

        answer = routine(callback, _INTEGER8_VALUE)
        assert answer == _INTEGER8_VALUE + _INTEGER8_INCREMENT
        assert observed == [_INTEGER8_VALUE]
