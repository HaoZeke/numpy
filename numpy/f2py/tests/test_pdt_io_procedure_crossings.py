"""Defined I/O and procedure calls preserve native parameter and owner state."""

import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestParameterizedDefinedIO(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types', 'pdt_io_crossings.f90')]

    @pytest.mark.parametrize('length', [0, 2, 5])
    def test_len_read_write_and_metadata(self, length):
        cls = self.module.pdt_io_primary.io_cell
        value = cls.from_string(f'{length} 17', n=length)
        assert value.value == 17
        assert str(value) == f'{length} 17'
        assert value.values.shape == (length,)
        np.testing.assert_array_equal(value.values, np.full(length, 17))

    def test_from_string_uses_declared_default_len(self):
        value = self.module.pdt_io_primary.io_cell.from_string('2 31')
        assert str(value) == '2 31'
        assert value.values.shape == (2,)

    def test_classmethod_preserves_python_subclass(self):
        class Child(self.module.pdt_io_primary.io_cell):
            pass

        value = Child.from_string('3 41', n=3)
        assert type(value) is Child
        assert value.value == 41
        assert str(value) == '3 41'
        np.testing.assert_array_equal(value.values, [41, 41, 41])

    def test_same_typename_modules_have_distinct_read_write_bindings(self):
        first = self.module.pdt_io_primary.io_cell.from_string('2 17', n=2)
        second = self.module.pdt_io_other.io_cell.from_string('1 1017', n=1)
        assert first.value == second.value == 17
        assert str(first) == '2 17'
        assert str(second) == '1 1017'
        assert type(first) is not type(second)

    def test_module_selectors_support_readonly_defined_output(self):
        module = self.module.pdt_io_primary
        module.reset_values()
        assert str(module.shared) == '3 31'
        assert str(module.locked) == '2 19'
        module.shared.value = 44
        assert str(module.shared) == '3 44'
        assert str(module.locked) == '2 19'

    @pytest.mark.parametrize('text', ['invalid', '3 17'])
    def test_native_read_error_preserves_other_values(self, text):
        cls = self.module.pdt_io_primary.io_cell
        held = cls.from_string('2 13', n=2)
        with pytest.raises(ValueError, match='read\\(formatted\\)'):
            cls.from_string(text, n=2)
        assert held.value == 13
        np.testing.assert_array_equal(held.values, [13, 13])

    def test_invalid_parameters_raise_without_native_default_fallback(self):
        cls = self.module.pdt_io_primary.io_cell
        with pytest.raises(ValueError):
            cls.from_string('2 17', n=-1)
        with pytest.raises(OverflowError):
            cls.from_string('2 17', n=2 ** 100)


@pytest.mark.slow
class TestParameterizedProcedurePointers(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_procedure_crossings.f90')]

    @pytest.mark.parametrize('length', [0, 2, 5])
    def test_owned_len_object_calls_both_pointer_forms(self, length):
        module = self.module.pdt_procedure_values
        obj = module.proc_cell(n=length, value=10)
        with pytest.raises(TypeError, match='not associated'):
            obj.compute(2.5)
        module.configure(obj)
        assert obj.compute(2.5) == 5
        assert obj.bump(3) is None
        assert obj.value == 13
        module.clear_compute(obj)
        with pytest.raises(TypeError, match='not associated'):
            obj.compute(1)

    def test_module_selection_and_protected_passed_object(self):
        module = self.module.pdt_procedure_values
        module.reset_values()
        shared = module.shared
        assert shared.compute(4) == 8
        shared.bump(5)
        assert module.shared.value == 12
        assert module.locked.compute(4) == 8
        with pytest.raises(AttributeError, match='read-only'):
            module.locked.bump(5)
        assert module.locked.value == 9

    @pytest.mark.parametrize('kind_name,configure,expected', [
        ('k_float', 'configure_float', 6), ('k_double', 'configure_double', 10)])
    def test_kind_identity_selects_native_component_layout(
        self, kind_name, configure, expected
    ):
        module = self.module.pdt_procedure_kinds
        obj = module.kind_proc(k=int(getattr(module, kind_name)))
        getattr(module, configure)(obj)
        assert obj.compute(2) == expected


@pytest.mark.slow
class TestOrdinaryProcedureConversionOwner(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types', 'proc_pointer_comp.f90')]

    def test_scalar_conversion_precedes_native_handle_extraction(self):
        obj = self.module.with_proc_ptr(id=1)
        self.module.proc_pointer_mod.set_func_square(obj)

        class Operand:
            def __float__(inner):
                obj.__init__(id=9)
                self.module.proc_pointer_mod.set_func_double(obj)
                return 3.0

        assert obj.func(Operand()) == 6
        assert obj.id == 9
