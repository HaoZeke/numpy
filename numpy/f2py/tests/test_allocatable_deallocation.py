"""Allocatable module data retains its allocation state across assignment."""

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


@pytest.mark.slow
class TestAllocatableDeallocation(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'common', 'allocatable_status.f90')]

    @pytest.mark.parametrize('name,value,index', [
        ('numbers', np.array([[1, 2], [3, 4]], dtype=np.int32), 0),
        ('words', np.array([b'ab', b'XYZ'], dtype='S3'), 1),
    ], ids=['integer-matrix', 'character-array'])
    def test_nonzero_storage_assignment_and_deallocation(self, name, value, index):
        module = self.module.allocation_status
        for _ in range(3):
            setattr(module, name, value)
            assert_array_equal(getattr(module, name), value)
            assert module.allocated_state()[index] == 1
            setattr(module, name, None)
            assert getattr(module, name) is None
            assert module.allocated_state()[index] == 0

    @pytest.mark.parametrize('shape', [(0, 3), (2, 0), (2, 3)])
    def test_zero_length_matrix_assignment_and_deallocation(self, shape):
        module = self.module.allocation_status
        value = np.ndarray(shape, dtype='S0', buffer=bytearray(1))
        assert value.itemsize == 0
        for _ in range(3):
            module.zeros = value
            assert module.zeros.dtype == np.dtype('S0')
            assert module.zeros.shape == shape
            assert module.zeros.tobytes() == b''
            assert module.allocated_state()[2] == 1
            module.zeros = None
            assert module.zeros is None
            assert module.allocated_state()[2] == 0
