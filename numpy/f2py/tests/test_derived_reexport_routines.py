"""Derived-array routines retain the owner of renamed and reexported types."""

import pytest

import numpy as np
from numpy.f2py import crackfortran, derived_type_rules
from numpy.f2py._dt_namespace import _find_type

from . import util

_SOURCE = util.getpath('tests', 'src', 'derived_types', 'reexport_routines.f90')


def _descriptions():
    modules = crackfortran.crackfortran([str(_SOURCE)])
    pymod = {'block': 'python module', 'name': 'reexport_probe', 'body': modules}
    return pymod, derived_type_rules.describe_types(pymod)


def test_reexport_descriptors_keep_original_owners():
    _, types = _descriptions()
    owner = types['reexport_owner', 'item']
    other = types['reexport_other_owner', 'item']
    assert owner['module'] == 'reexport_owner'
    assert owner['name'] == 'item'
    assert owner['c_name'] != other['c_name']
    assert ('reexport_facade', 'renamed') in owner['public_aliases']
    assert ('reexport_chain', 'chain_name') in owner['public_aliases']
    assert ('reexport_consumer', 'local') in owner['public_aliases']
    assert ('reexport_private_facade', 'hidden') not in owner['public_aliases']
    assert len(types) == 2
    assert len({id(info) for info in types.values()}) == 2


def test_direct_and_transitive_use_lookup_share_one_descriptor():
    _, types = _descriptions()
    owner = types['reexport_owner', 'item']
    direct = {'reexport_owner': {'only': 1, 'map': {'direct': 'item'}}}
    transitive = {'reexport_facade': {'only': 1, 'map': {'local': 'renamed'}}}
    chain = {'reexport_chain': {'only': 1, 'map': {'element': 'chain_name'}}}
    assert _find_type(types, 'direct', 'reexport_direct_consumer', direct) is owner
    assert _find_type(types, 'local', 'reexport_consumer', transitive) is owner
    assert _find_type(types, 'element', 'reexport_chain_consumer', chain) is owner
    assert _find_type(types, 'renamed', 'reexport_facade') is owner


def test_reexport_wildcard_respects_renames_and_private_exports():
    _, types = _descriptions()
    wildcard = {'reexport_facade': {'only': 0, 'map': {}}}
    hidden = {'reexport_private_facade': {'only': 1, 'map': {'local': 'hidden'}}}
    assert (_find_type(types, 'renamed', 'outside', wildcard)
            is types['reexport_owner', 'item'])
    assert _find_type(types, 'item', 'outside', wildcard) is None
    assert _find_type(types, 'local', 'outside', hidden) is None


def test_reexport_routine_scope_overrides_host_type():
    _, types = _descriptions()
    local = {'reexport_facade': {'only': 1, 'map': {'item': 'renamed'}}}
    assert (_find_type(types, 'item', 'reexport_other_owner', local_uses=local)
            is types['reexport_owner', 'item'])


def test_multiple_reexport_routes_do_not_duplicate_a_type():
    pymod, types = _descriptions()
    routes = {'reexport_facade': {'only': 1, 'map': {'local': 'renamed'}},
              'reexport_second_facade': {'only': 1, 'map': {'local': 'renamed'}}}
    assert (_find_type(types, 'local', 'outside', routes)
            is types['reexport_owner', 'item'])
    source = '\n'.join(derived_type_rules.buildhooks(pymod)['f90modhooks'])
    for info in types.values():
        assert source.count(f'static PyTypeObject Py{info["c_name"]}_Type = {{') == 1


@pytest.mark.slow
class TestReexportedDerivedArrayRoutines(util.F2PyTest):
    sources = [_SOURCE]

    def records(self, numbers):
        return [self.module.reexport_owner.item(value=n) for n in numbers]

    def test_direct_renamed_array_routine(self):
        records = self.records([2, 5, 8])
        assert self.module.reexport_direct_consumer.sum_direct(records) == 15

    def test_transitive_renamed_array_routine(self):
        records = self.records([2, 5, 8])
        assert self.module.reexport_consumer.sum_transitive(records) == 15

    def test_transitive_array_writeback(self):
        records = np.array(self.records([2, 5, 8]), dtype=object, order='F')
        self.module.reexport_consumer.bump_transitive(records, 4)
        assert [record.value for record in records] == [6, 9, 12]
        assert self.module.reexport_consumer.sum_transitive(records) == 27

    def test_multihop_rank_two_array_routine(self):
        records = np.array(self.records([2, 5, 8, 11]), dtype=object).reshape(2, 2)
        assert self.module.reexport_chain_consumer.sum_chain(records) == 26

    def test_routine_local_use_resolves_reexport(self):
        records = self.records([2, 5, 8])
        assert self.module.reexport_routine_local.sum_local(records) == 15

    def test_two_routes_accept_the_same_owning_class(self):
        a = self.records([2, 5])
        b = self.records([8, 11])
        assert self.module.reexport_two_routes.sum_two_routes(a, b) == 26

    def test_aliases_share_the_owning_class(self):
        owner = self.module.reexport_owner.item
        assert self.module.reexport_facade.renamed is owner
        assert self.module.reexport_chain.chain_name is owner
        assert self.module.reexport_consumer.local is owner
        assert self.module.reexport_other_owner.item is not owner

    def test_same_native_name_with_wrong_owner_is_rejected(self):
        wrong = [self.module.reexport_other_owner.item(value=2.0)]
        with pytest.raises(TypeError):
            self.module.reexport_consumer.sum_transitive(wrong)

    def test_empty_reexported_array(self):
        assert self.module.reexport_consumer.sum_transitive([]) == 0
