"""Public generic interfaces with inaccessible specific implementations."""

import pytest

from numpy.f2py import crackfortran, derived_type_rules

from . import util


class TestPrivateOperatorGeneration:
    def test_public_generics_use_accessible_expressions(self):
        path = util.getpath('tests', 'src', 'derived_types', 'private_operators.f90')
        modules = crackfortran.crackfortran([str(path)])
        module = modules[0]
        types = [node for node in module['body'] if node.get('block') == 'type']
        source = derived_type_rules.generate_fortran_wrappers(
            module['name'], types, module_block=module)
        assert 'source=default_add(' not in source
        assert 'call default_assign_integer(' not in source
        assert 'call default_assign_value(' not in source
        assert '((f2py_op_operand0) + (f2py_op_operand1))' in source
        assert 'f2py_op_operand0 = (f2py_op_arg1)' in source
        assert 'f2py_op_operand0 = (f2py_op_rhs)' in source
        hooks = derived_type_rules.buildhooks(module)
        generated = '\n'.join(hooks['f90modhooks'])
        assert '.nb_add =' in generated
        assert '.nb_negative =' in generated
        assert '.nb_multiply =' not in generated


@pytest.mark.slow
class TestPrivateOperatorInterfaces(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types', 'private_operators.f90')]

    def test_default_private_binary_and_unary(self):
        left = self.module.private_value(value=3)
        right = self.module.private_value(value=8)
        assert (left + right).value == 11
        assert (-left).value == -3
        assert left.value == 3
        assert right.value == 8

    def test_default_private_comparison(self):
        value = self.module.private_value(value=3)
        assert value == self.module.private_value(value=3)
        assert value != self.module.private_value(value=4)

    def test_default_private_assignment_uses_actual_destination(self):
        destination = self.module.private_value(value=3)
        assert destination.assign_from(8) is destination
        assert destination.value == 11

    def test_default_private_record_and_self_assignment(self):
        destination = self.module.private_value(value=3)
        source = self.module.private_value(value=8)
        assert destination.assign_from(source) is destination
        assert destination.value == 9
        assert source.value == 8
        assert destination.assign_from(destination) is destination
        assert destination.value == 10

    def test_private_generic_has_no_number_slot(self):
        value = self.module.private_value(value=3)
        with pytest.raises(TypeError):
            value * value

    def test_public_procedure_remains_callable(self):
        value = self.module.private_value(value=3)
        assert self.module.default_private_operators.public_value_total(value) == 3

    def test_explicit_private_implementation(self):
        left = self.module.explicit_value(value=3)
        right = self.module.explicit_value(value=8)
        assert (left + right).value == 11
        assert left.assign_from(8) is left
        assert left.value == 16
        assert right.value == 8

    def test_explicit_private_operator_has_no_unary_slot(self):
        value = self.module.explicit_value(value=3)
        with pytest.raises(TypeError):
            -value
