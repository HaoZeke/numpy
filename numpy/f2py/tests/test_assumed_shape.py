import os
import tempfile

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from . import util


# Note: do not mark as slow! This is one of a small set of f2py compile tests retained
# in the default suite invocation for compile-path coverage.
class TestAssumedShapeSumExample(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "assumed_shape", "foo_free.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_use.f90"),
        util.getpath("tests", "src", "assumed_shape", "precision.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_mod.f90"),
        util.getpath("tests", "src", "assumed_shape", ".f2py_f2cmap"),
    ]

    def test_all(self):
        r = self.module.fsum([1, 2])
        assert r == 3
        r = self.module.sum([1, 2])
        assert r == 3
        r = self.module.sum_with_use([1, 2])
        assert r == 3

        r = self.module.mod.sum([1, 2])
        assert r == 3
        r = self.module.mod.fsum([1, 2])
        assert r == 3


@pytest.mark.slow
class TestF2cmapOption(TestAssumedShapeSumExample):
    def setup_method(self):
        # Use a custom file name for .f2py_f2cmap
        self.sources = list(self.sources)
        f2cmap_src = self.sources.pop(-1)

        self.f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f2cmap_src, "rb") as f:
            self.f2cmap_file.write(f.read())
        self.f2cmap_file.close()

        self.sources.append(self.f2cmap_file.name)
        self.options = ["--f2cmap", self.f2cmap_file.name]

        super().setup_method()

    def teardown_method(self):
        os.unlink(self.f2cmap_file.name)


@pytest.mark.slow
class TestAssumedShapeLowerBound(util.F2PyTest):
    # gh-25421: assumed-shape dummies keep their declared lower bounds
    sources = [util.getpath("tests", "src", "assumed_shape", "gh25421.f90")]

    def test_bounds(self):
        lo, hi = self.module.gh25421.bounds(np.zeros(5, dtype=np.complex128))
        assert (lo, hi) == (0, 4)
        lo, hi = self.module.ext_bounds(np.zeros(3))
        assert (lo, hi) == (0, 2)

    def test_inout(self):
        b = np.zeros((2, 3), order="F")
        self.module.gh25421.fill(b)
        i, j = np.meshgrid(np.arange(-1, 1), np.arange(2, 5), indexing="ij")
        assert_array_equal(b, 10 * i + j)

    def test_function(self):
        assert self.module.gh25421.total(np.arange(4.0)) == 6.0
