"""Module-qualified identities for colliding derived type declarations."""

import pytest

from numpy.f2py import crackfortran, derived_type_rules
from numpy.f2py._dt_namespace import _find_type, _qualify_fortran

from . import util

_SOURCE = util.getpath('tests', 'src', 'derived_types', 'qualified_names.f90')


def _descriptions():
    modules = crackfortran.crackfortran([str(_SOURCE)])
    pymod = {'block': 'python module', 'name': 'namespace_probe', 'body': modules}
    return pymod, derived_type_rules.describe_types(pymod)


def test_distinct_declarations_preserve_native_names():
    pymod, types = _descriptions()
    alpha = types['namespace_alpha', 'item']
    beta = types['namespace_beta', 'item']
    assert alpha['name'] == beta['name'] == 'item'
    assert alpha['c_name'] != beta['c_name']
    assert alpha['bindc'] and not beta['bindc']
    derived_type_rules.buildhooks(pymod)
    assert alpha['block']['name'] == beta['block']['name'] == 'item'


def test_type_lookup_follows_declaring_module():
    _, types = _descriptions()
    assert (_find_type(types, 'item', 'namespace_alpha')
            is types['namespace_alpha', 'item'])
    assert (_find_type(types, 'item', 'namespace_beta')
            is types['namespace_beta', 'item'])
    assert _find_type(types, 'item') is None


def test_type_lookup_follows_use_rename():
    _, types = _descriptions()
    uses = {'namespace_alpha': {'only': 1, 'map': {'alpha_item': 'item'}},
            'namespace_beta': {'only': 1, 'map': {'beta_item': 'item'}}}
    assert (_find_type(types, 'alpha_item', 'namespace_join', uses)
            is types['namespace_alpha', 'item'])
    assert (_find_type(types, 'beta_item', 'namespace_join', uses)
            is types['namespace_beta', 'item'])
    assert _find_type(types, 'item', 'namespace_join', uses) is None


def test_routine_use_overrides_host_type():
    _, types = _descriptions()
    local = {'namespace_beta': {'only': 1, 'map': {'item': 'item'}}}
    assert (_find_type(types, 'item', 'namespace_alpha', local_uses=local)
            is types['namespace_beta', 'item'])


@pytest.mark.slow
class TestQualifiedDerivedTypes(util.F2PyTest):
    sources = [_SOURCE]

    def test_native_module_type_aliases_are_distinct(self):
        alpha = self.module.namespace_alpha.item
        beta = self.module.namespace_beta.item
        assert alpha is not beta
        assert self.module.namespace_alpha_item is alpha
        assert self.module.namespace_beta_item is beta

    def test_bindc_result_uses_its_own_layout(self):
        obj = self.module.namespace_alpha.make_item(2)
        assert isinstance(obj, self.module.namespace_alpha.item)
        assert obj.value == 2
        assert self.module.namespace_alpha.read_item(obj) == 2

    def test_opaque_result_uses_its_own_storage(self):
        obj = self.module.namespace_beta.make_item(40.0)
        assert isinstance(obj, self.module.namespace_beta.item)
        assert obj.value == 40.0
        assert self.module.namespace_beta.read_item(obj) == 40.0

    def test_two_renamed_imports_reach_their_own_types(self):
        a = self.module.namespace_alpha.make_item(2)
        b = self.module.namespace_beta.make_item(40.0)
        assert self.module.namespace_join.total(a, b) == 42.0

    def test_wrong_module_type_is_rejected(self):
        obj = self.module.namespace_beta.make_item(40.0)
        with pytest.raises(TypeError):
            self.module.namespace_alpha.read_item(obj)


@pytest.mark.parametrize('verb', ['resize', 'clear'])
def test_array_binding_labels_follow_module_identity(verb):
    pymod, types = _descriptions()
    labels = []
    for module in pymod['body']:
        if module['name'] not in ('namespace_alpha', 'namespace_beta'):
            continue
        info = types[module['name'], 'item']
        native = f'f2py_{verb}_item_values'
        source = (f'subroutine {native}(obj) bind(c)\n'
                  '  type(item), pointer :: obj\n'
                  f'end subroutine {native}\n')
        generated = _qualify_fortran(source, module)
        label = f"f2py_{verb}_{info['c_name']}_values"
        assert generated == source.replace('bind(c)', f'bind(c, name="{label}")')
        assert 'type(item), pointer :: obj' in generated
        labels.append(label)
    assert len(labels) == 2
    assert labels[0] != labels[1]


def test_absent_parent_terminates_type_lookup():
    assert _find_type({}, None) is None


def test_qualified_operator_binding_uses_free_form_continuation():
    from numpy.f2py._dt_namespace import _qualify_fortran

    module = {'f2py_dt_namespace': {'item': 'f2pyq_' + 'a' * 24}}
    source = (
        '  function f2py_op_item_namespace_item_sum(f2py_op_arg0, '
        'f2py_op_arg1) result(f2py_op_result) bind(c)\n'
        '    type(item) :: native_value\n'
        '  end function f2py_op_item_namespace_item_sum\n')
    result = _qualify_fortran(source, module)
    assert all(len(line) <= 132 for line in result.splitlines())
    assert ' &\n    & bind(c, name="f2py_op_f2pyq_' in result
    assert 'type(item) :: native_value' in result
    assert 'end function f2py_op_item_namespace_item_sum' in result


@pytest.mark.slow
class TestQualifiedOperatorBindings(util.F2PyTest):
    sources = [util.getpath("tests", "src", "derived_types", "qualified_names.f90")]

    def test_qualified_operator_and_assignment_bindings(self):
        item = self.module.namespace_alpha.item
        a, b = item(value=2), item(value=5)
        result = a + b
        assert result.value == 7
        assert type(result) is item
        assert a.__fortran_assign__(11) is a
        assert a.value == 14


def test_native_component_and_literal_names_are_preserved():
    pymod, types = _descriptions()
    module = next(m for m in pymod['body'] if m['name'] == 'namespace_beta')
    info = types['namespace_beta', 'item']
    source = ('subroutine f2py_get_item_value(obj) bind(c)\n'
              '  type(item), pointer :: obj\n'
              '  obj%f2py_get_item_value = 3\n'
              '  print *, "f2py_get_item_value"\n'
              'end subroutine f2py_get_item_value\n')
    label = f"f2py_get_{info['c_name']}_value"
    assert _qualify_fortran(source, module) == source.replace(
        'bind(c)', f'bind(c, name="{label}")')


@pytest.mark.slow
class TestQualifiedNamespaceBoundaries(util.F2PyTest):
    sources = [_SOURCE]

    def test_native_component_with_generated_prefix(self):
        obj = self.module.namespace_beta.make_item(40.0)
        obj.f2py_get_item_value = 2.0
        assert obj.f2py_get_item_value == 2.0
        assert self.module.namespace_beta.read_shadow(obj) == 2.0

    def test_polymorphic_family_with_unrelated_root(self):
        module = self.module.namespace_poly
        base = module.base(value=2.0)
        child = module.child(value=3.0, extra=4.0)
        assert module.read_base(base) == 2.0
        assert module.read_base(child) == 3.0
