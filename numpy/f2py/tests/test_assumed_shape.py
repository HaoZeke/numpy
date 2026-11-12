import os
import tempfile

import pytest

import numpy as np

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
class TestAssumedShapeFunctionResult(util.F2PyTest):
    # The function name is the result variable when there is no
    # result clause.
    sources = [
        util.getpath("tests", "src", "assumed_shape", "function_result.f90"),
    ]

    def test_function_result(self):
        assert self.module.sum_noresult([1, -2, 3]) == 2
        assert self.module.count_noresult([1, -2, 3]) == 2


@pytest.mark.slow
class TestAssumedShapeSignature(util.F2PyTest):
    # gh-13553: f2py-only syntax of a signature file must not reach the
    # interface in the Fortran wrapper
    sources = [
        util.getpath("tests", "src", "assumed_shape", "gh13553.f90"),
        util.getpath("tests", "src", "assumed_shape", "gh13553.pyf"),
    ]
    module_name = "gh13553"

    def test_generated_signature(self):
        assert self.module.trapz([0, 1, 2], [0, 1, 4]) == 3
        with pytest.raises(ValueError, match="2nd argument `y`"):
            self.module.trapz([0, 1, 2], [0, 1])
        a = np.ones(4)
        self.module.scale_by(a, [2, 3])
        assert a.tolist() == [2, 3, 1, 1]

    def test_written_signature(self):
        assert self.module.fill(np.zeros(3), [1, 2]).tolist() == [1, 2, 0]
        assert self.module.dsum([1, 2, 3]) == 6


@pytest.mark.slow
class TestAssumedShapeDirective(util.F2PyTest):
    # gh-13553: the same holds for f2py directives in a Fortran source
    sources = [util.getpath("tests", "src", "assumed_shape", "gh13553.f90")]

    def test_hidden_argument(self):
        a = np.zeros(3)
        self.module.fill_hidden(a, [1, 2])
        assert a.tolist() == [1, 2, 0]


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
