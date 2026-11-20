import os
import tempfile

import pytest

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
class TestAssumedShapeFunctionType(util.F2PyTest):
    suffix = ".f90"
    code = """
integer function count_items(b)
  double precision, intent(in) :: b(:)
  count_items = size(b)
end function

double precision function idx_sum(b)
  double precision, intent(in) :: b(:)
  idx_sum = sum(b)
end function

logical function has_items(b)
  double precision, intent(in) :: b(:)
  has_items = size(b) > 0
end function

integer function count_result(b) result(r)
  double precision, intent(in) :: b(:)
  r = size(b)
end function
"""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("count_items", 3),
            ("idx_sum", 6.0),
            ("has_items", 1),
            ("count_result", 3),
        ],
    )
    def test_function_return_type(self, name, expected):
        assert getattr(self.module, name)([1.0, 2.0, 3.0]) == expected
