"""Array-valued functions require their result rank at the Fortran call site."""

import copy

import pytest

import numpy as np
from numpy.f2py import func2subr
from numpy.testing import assert_array_equal

from . import util


def _routine(variable, declaration, *, module=False):
    result = copy.deepcopy(variable)
    result['intent'] = ['out']
    argument = copy.deepcopy(variable)
    argument['intent'] = ['in']
    interface = (
        "function array_result(value) result(output)\n"
        f"        {declaration.replace('output', 'value')}\n"
        f"        {declaration}\n"
        "      end function array_result"
    )
    routine = {
        'block': 'function',
        'name': 'array_result',
        'args': ['array_result', 'value'],
        'vars': {'array_result': result, 'value': argument},
        'saved_interface': interface,
    }
    if module:
        routine['modulename'] = 'array_results'
    return routine


@pytest.mark.parametrize('variable, declaration', [
    ({'typespec': 'character', 'dimension': ['3']}, 'character output(3)'),
    ({'typespec': 'character', 'charselector': {'len': '3'}, 'dimension': ['3']},
     'character(len=3) output(3)'),
    ({'typespec': 'real', 'kindselector': {'kind': '8'}, 'dimension': ['3']},
     'real(kind=8) output(3)'),
    ({'typespec': 'integer', 'dimension': ['2', '3']}, 'integer output(2,3)'),
])
def test_external_array_result_has_explicit_interface(variable, declaration):
    wrapper = func2subr.createfuncwrapper(_routine(variable, declaration))
    assert '\n      interface\n' in wrapper
    assert 'function array_result(value) result(output)' in wrapper
    assert declaration in wrapper
    assert '\n      end interface\n' in wrapper
    assert 'external array_result' not in wrapper
    assert 'array_resultf2pywrap = array_result(value)' in wrapper


def test_module_array_result_uses_module_interface():
    variable = {'typespec': 'character', 'dimension': ['3']}
    wrapper = func2subr.createfuncwrapper(
        _routine(variable, 'character output(3)', module=True)
    )
    assert 'use array_results, only : array_result' in wrapper
    assert '\n      interface\n' not in wrapper
    assert 'external array_result' not in wrapper


@pytest.mark.parametrize('variable, declaration', [
    ({'typespec': 'character'}, 'character output'),
    ({'typespec': 'real'}, 'real output'),
])
def test_scalar_result_keeps_external_declaration(variable, declaration):
    wrapper = func2subr.createfuncwrapper(_routine(variable, declaration))
    assert 'external array_result' in wrapper
    assert '\n      interface\n' not in wrapper


@pytest.mark.slow
class TestCharacterArrayResults(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'return_character', 'array_return.f90')]

    @pytest.mark.parametrize('routine, value', [
        ('fixed_letters', np.array(list('abc'), dtype='S1')),
        ('fixed_strings', np.array([b'abc', b'def', b'ghi'], dtype='S3')),
        ('letter_matrix', np.array([list('abc'), list('def')], dtype='S1', order='F')),
        ('numeric_array', np.array([1.25, -2.5, 4.0], dtype=np.float64)),
    ])
    def test_external_result(self, routine, value):
        result = getattr(self.module, routine)(value)
        assert_array_equal(result, value)
        assert result.shape == value.shape
        assert result.dtype == value.dtype

    def test_module_result(self):
        value = np.array(list('xyz'), dtype='S1')
        result = self.module.module_array_results.module_letters(value)
        assert_array_equal(result, value)
        assert result.dtype == value.dtype

    def test_results_own_their_storage(self):
        values = [np.array(list(text), dtype='S1') for text in ('abc', 'xyz', '123')]
        results = [self.module.fixed_letters(value) for value in values]
        for value, result in zip(values, results):
            assert_array_equal(result, value)
        results[-1][0] = b'!'
        assert_array_equal(results[0], values[0])
        assert_array_equal(results[1], values[1])
