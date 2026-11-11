import threading

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


@pytest.mark.slow
class TestAllocatableArguments(util.F2PyTest):
    sources = [util.getpath("tests", "src", "allocatable_args", "alloc_args.f90")]

    def test_intent_out(self):
        # gh-19157
        x = self.module.alloc_out(2, 3)
        assert x.dtype == np.float64
        assert x.flags.f_contiguous
        assert_array_equal(x, 10 * np.arange(3)[:, None] + np.arange(4))

    def test_intent_out_unallocated(self):
        assert self.module.maybe_out(-1) is None
        x = self.module.maybe_out(0)
        assert x.shape == (0,)
        assert x.dtype == np.intc
        assert_array_equal(self.module.maybe_out(3), [3, 3, 3])

    def test_intent_out_module(self):
        assert_array_equal(self.module.alloc_args.mod_out(3), [1.0, 2.0, 3.0])

    def test_intent_out_threadsafe(self):
        # the routine runs with the GIL released
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    self.module.alloc_args.threadsafe_out(2)))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4
        for x in results:
            assert x.shape == (2, 3, 2)
            assert_array_equal(x, np.full((2, 3, 2), 1 - 1j))

    def test_intent_in(self):
        a = np.arange(6.0).reshape(2, 3)
        assert_array_equal(self.module.alloc_in(a), [15.0, 2.0, 3.0])
        assert_array_equal(self.module.alloc_in(a[:, ::2]), [10.0, 2.0, 2.0])

    def test_intent_inout(self):
        x = np.arange(4, dtype=np.intc)
        assert self.module.alloc_inout(x) is None
        assert_array_equal(x, [0, 2, 4, 6])

    def test_intent_inout_reallocated(self):
        x = np.arange(4, dtype=np.intc)
        with pytest.raises(ValueError, match="reallocated"):
            self.module.alloc_grow(x)
        assert_array_equal(x, [0, 1, 2, 3])

    def test_intent_in_out(self):
        x = self.module.alloc_args.append([1, 2], 3)
        assert_array_equal(x, [1, 2, -1, -1, -1])

    def test_function_argument(self):
        k, x = self.module.count_out(3)
        assert k == 3
        assert x.dtype == np.float32
        assert_array_equal(x, [0.5, 0.5, 0.5])
        k, x = self.module.alloc_args.mod_count(2)
        assert k == 2
        assert_array_equal(x, [7, 7])

    def test_function_result(self):
        assert_array_equal(self.module.arange(4), [0, 1, 2, 3])
        assert_array_equal(self.module.alloc_args.mod_range(3), [1, 2, 3])
