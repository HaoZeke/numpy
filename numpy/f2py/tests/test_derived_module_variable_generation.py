"""Selectors preserve native ownership, privacy, and parameter identities."""

from pathlib import Path

import pytest

from numpy.f2py import crackfortran, derived_type_rules
from numpy.f2py._dt_module_variables import (
    build_module_variable_hooks,
    generate_module_variable_fortran,
)

from . import util


def prepared():
    path = util.getpath('tests', 'src', 'derived_types',
                        'pdt_module_variables.f90')
    modules = crackfortran.crackfortran([path])
    pymod = {'block': 'python module', 'name': 'module_state', 'body': modules}
    descriptions = derived_type_rules.describe_types(pymod)
    return modules, descriptions


class TestDerivedModuleVariableGeneration:
    def test_canonical_descriptor_is_not_duplicated(self):
        modules, descriptions = prepared()
        assert len(descriptions) == 1
        descriptor = next(iter(descriptions.values()))
        records = descriptor['block']['f2py_pdt_bindings']
        assert len(records) == 10
        assert len({record['id'] for record in records}) == 10
        assert {record['variable'] for record in records} == {
            'single', 'fixed', 'cube', 'optional', 'allocated_cells',
            'pool', 'selected', 'locked', 'renamed_single', 'fixed_allocated'}
        assert all(record['c_name'] == descriptor['c_name'] for record in records)
        assert all(record['rank'] <= 3 for record in records)
        assert not any(record['variable'] == 'hidden' for record in records)
        assert len(modules[0]['f2py_pdt_bindings']) == 9
        assert all(record in records for record in modules[0]['f2py_pdt_bindings'])

    def test_non_target_selections_never_export_an_address(self):
        modules, _ = prepared()
        source = generate_module_variable_fortran(modules)
        for record in modules[0]['f2py_pdt_bindings']:
            assert f'c_loc({record["alias"]})' not in source
            assert f'=> {record["alias"]}' not in source
            assert f'c_funloc({record["prefix"]}_valid)' in source
        assert 'header%storage /= 1' in source
        assert 'kind=c_intptr_t' in source

    @pytest.mark.parametrize('name', ['optional', 'allocated_cells', 'selected'])
    def test_availability_precedes_shape_and_member_access(self, name):
        modules, _ = prepared()
        record = next(item for item in modules[0]['f2py_pdt_bindings']
                      if item['variable'] == name)
        source = generate_module_variable_fortran(modules)
        start = source.index(f'function {record["prefix"]}_valid')
        end = source.index(f'end function {record["prefix"]}_valid', start)
        validator = source[start:end]
        check = 'allocated' if record['allocatable'] else 'associated'
        assert validator.index(f'{check}({record["alias"]})') < validator.index(
            'call c_f_pointer(header%lengths')
        assert 'header%length_count /= 1' in validator

    def test_setters_copy_sources_before_mutation(self):
        modules, _ = prepared()
        source = generate_module_variable_fortran(modules)
        for record in modules[0]['f2py_pdt_bindings']:
            if record['readonly']:
                assert f'subroutine {record["prefix"]}_write' not in source
                continue
            start = source.index(f'subroutine {record["prefix"]}_write')
            end = source.index(f'end subroutine {record["prefix"]}_write', start)
            setter = source[start:end]
            commit = ('call move_alloc(temp, ' + record['alias']
                      if record['allocatable'] else '= copies(position)%value')
            assert (setter.index('f2py_pdt_validate(handles(position))')
                    < setter.index(commit))
            assert (setter.index('allocate(copies(position)%value,')
                    < setter.index(commit))
        assert all(len(line) <= 132 for line in source.splitlines())

    def test_c_descriptors_pin_capsules_and_own_the_module(self):
        modules, _ = prepared()
        hooks = build_module_variable_hooks(modules)
        source = '\n'.join(hooks['f90modhooks'])
        initialization = '\n'.join(hooks['initf90modhooks'])
        assert 'Py_XINCREF(pins[i])' in source
        records = modules[0]['f2py_pdt_bindings']
        assert all(f'PyCapsule_GetPointer(pins[i], "{item["capsule"]}")' in source
                   for item in records if not item['readonly'])
        assert 'Py_XDECREF(owner)' in source
        assert 'F2Py_SetModuleGetSet' in initialization
        assert 'PyObject_SetAttrString(owner' not in source

    def test_fixture_declares_non_target_and_protected_state(self):
        source = Path(util.getpath('tests', 'src', 'derived_types',
                                  'pdt_module_variables.f90')).read_text()
        assert 'type(cell(c_double, 2)) :: single' in source
        assert 'type(cell(c_double, 2)) :: fixed(-1:0, 3:5)' in source
        assert 'type(cell(c_double, 2)), protected :: locked' in source
