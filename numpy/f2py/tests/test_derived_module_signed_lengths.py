"""Signed LEN metadata preserves native zero extents and module identity."""

import pytest

from numpy.f2py import crackfortran, derived_type_rules
from numpy.f2py._dt_module_variables import (
    _length_checks,
    generate_module_variable_fortran,
    prepare_module_variable_bindings,
)

from . import util


class TestModuleSignedLengthGeneration:
    def test_length_conversion_checks_both_c_int_bounds(self):
        record = {'description': {'lengths': [{'name': 'n'}]}}
        source = '\n'.join(_length_checks(record, 'value', 'lengths', 'return'))
        assert '-int(huge(0_c_int), c_int64_t)-1_c_int64_t' in source
        assert 'int(huge(0_c_int), c_int64_t)) return' in source
        assert 'if (value%n < 0)' not in source
        assert 'lengths(1) = int(value%n, c_int)' in source

    def test_setter_uses_status_separately_from_signed_value(self):
        path = util.getpath('tests', 'src', 'derived_types',
                            'pdt_signed_module_lengths.f90')
        modules = crackfortran.crackfortran([path])
        pymod = {'block': 'python module', 'name': 'signed_state', 'body': modules}
        descriptions = derived_type_rules.describe_types(pymod)
        prepare_module_variable_bindings(modules, descriptions)
        source = generate_module_variable_fortran(modules)
        for record in modules[0]['f2py_pdt_bindings']:
            start = source.index(f'subroutine {record["prefix"]}_write')
            end = source.index(f'end subroutine {record["prefix"]}_write', start)
            setter = source[start:end]
            assert 'f2py_pdt_length_value(handles(position), 0,' in setter
            assert 'current_length) /= 0) return' in setter
            assert 'if (current_length /= lengths(1)) return' in setter
            assert 'lengths(1) < 0' not in setter
            assert 'f2py_pdt_length(handles' not in setter


@pytest.mark.slow
class TestModuleSignedLengths(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_signed_module_lengths.f90')]

    @pytest.mark.parametrize('which,name,length',
                             [(1, 'minus_one', -1), (3, 'minus_three', -3)])
    def test_fixed_selector_preserves_negative_len(self, which, name, length):
        module = self.module.pdt_signed_module_lengths
        module.reset_values()
        value = getattr(module, name)
        assert value.n == length
        assert value.values.shape == (0,)
        assert module.native_length(which) == length
        value.value = 41
        assert module.native_value(which) == 41

    @pytest.mark.parametrize('which,name,length',
                             [(1, 'minus_one', -1), (3, 'minus_three', -3)])
    def test_assignment_preserves_signed_source_lengths(self, which, name, length):
        module = self.module.pdt_signed_module_lengths
        module.reset_values()
        incoming = self.module.signed_cell(n=length, value=31)
        setattr(module, name, incoming)
        assert getattr(module, name).n == length
        assert module.native_length(which) == length
        assert module.native_value(which) == 31
        assert getattr(module, name).values.shape == (0,)

    @pytest.mark.parametrize('length', [-1, -3])
    def test_allocatable_selector_refreshes_signed_lengths(self, length):
        module = self.module.pdt_signed_module_lengths
        module.reset_values()
        module.allocate_optional(2)
        held = module.optional
        assert held.n == 2
        module.allocate_optional(length)
        assert held.n == length
        assert held.values.shape == (0,)
        assert held.value == 17
