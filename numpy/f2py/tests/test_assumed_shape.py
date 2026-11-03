import os
import tempfile

import pytest

import numpy as np
from numpy.f2py import crackfortran
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


LOWER_BOUND_SRC = util.getpath("tests", "src", "assumed_shape", "lower_bound.f90")


def test_lower_bound_is_assumed_shape():
    # gh-25421: an explicit lower bound such as x(0:) produced the
    # dimension "1 + ", which ended up verbatim in the generated C
    mod = crackfortran.crackfortran([str(LOWER_BOUND_SRC)])
    routines = {b["name"]: b["vars"] for b in mod[0]["body"]}
    assert routines["bounds"]["x"]["dimension"] == [":"]
    assert routines["bounds_2d"]["x"]["dimension"] == [":", ":"]
    assert routines["add_index"]["x"]["dimension"] == [":"]
    assert routines["first_sum"]["a"]["dimension"] == [":"]
    assert routines["first_sum"]["c"]["dimension"] == [":"]


@pytest.mark.slow
class TestAssumedShapeLowerBound(util.F2PyTest):
    sources = [LOWER_BOUND_SRC]

    def test_bounds(self):
        lb, ub = self.module.lower_bound.bounds(np.arange(5.0))
        assert (lb, ub) == (0, 4)

    def test_bounds_2d(self):
        lb, ub = self.module.lower_bound.bounds_2d(np.zeros((2, 3), order="F"))
        assert_array_equal(lb, [0, -1])
        assert_array_equal(ub, [1, 1])

    def test_variable_lower_bound(self):
        x = np.zeros(4)
        self.module.lower_bound.add_index(x, 3)
        assert_array_equal(x, [3, 4, 5, 6])

    def test_function(self):
        r = self.module.lower_bound.first_sum([1 + 2j, 5], [3 + 4j])
        assert r == 4 + 6j
