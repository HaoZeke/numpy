"""Negative LEN metadata is valid for zero-extent derived components."""

import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestDefinedIONegativeLengths(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'derived_types', 'pdt_io_crossings.f90')]

    @pytest.mark.parametrize('length', [-1, -3])
    def test_read_retains_negative_len_and_zero_extent(self, length):
        cls = self.module.pdt_io_primary.io_cell
        value = cls.from_string(f'{length} 17', n=length)
        assert value.value == 17
        assert value.n == length
        assert value.values.shape == (0,)
        assert str(value) == f'{length} 17'
        np.testing.assert_array_equal(value.values, np.empty(0, dtype=int))
