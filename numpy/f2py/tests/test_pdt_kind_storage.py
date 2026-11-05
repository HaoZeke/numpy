"""KIND-only native handle storage and numerical components."""

import pytest

import numpy as np
from numpy.f2py._dt_handles import _handle_description, _handle_module

from . import util


def _type(length=False):
    variables = {'k': {'typespec': 'integer', 'attrspec': ['kind'], '=': '4'}}
    if length:
        variables['n'] = {'typespec': 'integer', 'attrspec': ['len']}
    return {'block': 'type', 'name': 'cell', 'vars': variables,
            'parent_block': {'block': 'module', 'name': 'cells'}}


def test_kind_only_target_has_no_nested_parameterized_component():
    description = _handle_description(_type())
    source = _handle_module(description)
    assert f'type :: {description["box"]}' not in source
    assert 'c_loc(value)' in source
    assert 'call c_f_pointer(header%payload, value)' in source
    assert 'c_loc(box)' not in source


def test_len_target_retains_its_native_descriptor_holder():
    description = _handle_description(_type(length=True))
    source = _handle_module(description)
    assert f'type :: {description["box"]}' in source
    assert 'c_loc(box)' in source
    assert 'call c_f_pointer(header%payload, box)' in source
    assert 'c_loc(value)' not in source


@pytest.mark.slow
class TestKindNativeStorage(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types',
                            'kind_array_boundaries.f90')]

    @pytest.mark.parametrize('kind,dtype', [(4, np.float32), (8, np.float64)])
    def test_instances_keep_independent_native_components(self, kind, dtype):
        first = self.module.kind_array(k=kind)
        second = self.module.kind_array(k=kind)
        positive = (np.arange(6, dtype=dtype).reshape(2, 3) + dtype(0.125))
        negative = -positive
        first.plane = positive
        second.plane = negative
        np.testing.assert_array_equal(first.plane, positive)
        np.testing.assert_array_equal(second.plane, negative)
        cube = np.arange(24, dtype=dtype).reshape(2, 3, 4) + dtype(0.25)
        first.cube = cube
        assert second.cube is None
        np.testing.assert_array_equal(first.cube, cube)
        first.cube = None
        second.cube = -cube
        assert first.cube is None
        np.testing.assert_array_equal(second.cube, -cube)

    @pytest.mark.parametrize('kind,dtype', [(4, np.float32), (8, np.float64)])
    def test_component_conversion_uses_selected_native_kind(self, kind, dtype):
        value = self.module.kind_array(k=kind)
        source = np.array([0.1, 0.2, 0.3, -0.7, 1.1, 2.9]).reshape(2, 3)
        value.plane = source
        assert value.k == kind
        assert value.plane.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(value.plane, source.astype(dtype))
