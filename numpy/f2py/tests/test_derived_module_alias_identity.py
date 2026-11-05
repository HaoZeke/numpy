"""Export names preserve the identity of their actual native module state."""

import pytest

from numpy.f2py._dt_handles import _handle_description
from numpy.f2py._dt_module_variables import _canonical_bindings

from . import util
from .test_derived_module_variable_generation import prepared


class TestModuleVariableStorageIdentity:
    def test_export_records_share_one_declaring_storage_id(self):
        _, types = prepared()
        block = next(iter(types.values()))['block']
        records = block['f2py_pdt_bindings']
        original = next(item for item in records
                        if item['module'] == 'pdt_module_values'
                        and item['variable'] == 'single')
        facade = next(item for item in records
                      if item['module'] == 'pdt_module_facade')
        assert original['id'] != facade['id']
        assert original['storage_id'] == facade['storage_id']
        assert len({item['id'] for item in records}) == len(records)
        assert len({item['storage_id'] for item in records}) == len(records) - 1

    def test_native_dispatch_has_one_case_per_storage_slot(self):
        _, types = prepared()
        block = next(iter(types.values()))['block']
        records = block['f2py_pdt_bindings']
        projected = _canonical_bindings(records)
        assert len(projected) == len(records) - 1
        assert len({item['id'] for item in projected}) == len(projected)
        assert all(item['id'] == item['storage_id'] for item in projected)
        assert all(item['module'] == item['origin_module'] for item in projected)

    def test_handle_descriptions_use_canonical_dispatch_ids(self):
        _, types = prepared()
        block = next(iter(types.values()))['block']
        records = block['f2py_pdt_bindings']
        identities = {item['identity_number'] for item in records}
        for identity in identities:
            original = next(item for item in records
                            if item['identity_number'] == identity)
            spec = original['description']['pointer_spec']
            description = _handle_description(block, spec)
            cases = description['bindings']
            assert len({item['id'] for item in cases}) == len(cases)
            expected = {item['storage_id'] for item in records
                        if item['identity_number'] == identity}
            assert {item['id'] for item in cases} == expected


@pytest.mark.slow
class TestNativeModuleAliasIdentity(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'pdt_module_alias_identity.f90')]

    @pytest.mark.parametrize('scope,name', [
        ('pdt_alias_facade', 'renamed'), ('pdt_alias_consumer', 'local')])
    def test_reexport_self_assignment_preserves_rhs(self, scope, name):
        owner = self.module.pdt_alias_owner
        owner.reset_values()
        lhs = owner.primary
        rhs = getattr(getattr(self.module, scope), name)
        returned = lhs.__fortran_assign__(rhs)
        assert returned is lhs
        assert owner.primary.value == 37
        assert rhs.value == 37

    def test_distinct_target_and_pointer_bindings_detect_actual_alias(self):
        owner = self.module.pdt_alias_owner
        owner.reset_values()
        lhs = owner.selected[1]
        rhs = owner.pool[1]
        returned = lhs.__fortran_assign__(rhs)
        assert returned is lhs
        assert owner.pool[0].value == 11
        assert owner.pool[1].value == 43
        assert owner.selected[1].value == 43
        owner.clear_pointer()

    def test_distinct_cells_remain_distinct_actuals(self):
        owner = self.module.pdt_alias_owner
        owner.reset_values()
        lhs = owner.pool[0]
        rhs = owner.selected[1]
        lhs.__fortran_assign__(rhs)
        assert owner.pool[0].value == 43
        assert owner.pool[1].value == 13
        owner.clear_pointer()
