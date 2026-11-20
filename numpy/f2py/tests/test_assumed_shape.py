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
class TestInterfaceBoundDeclOrder(util.F2PyTest):
    suffix = ".f90"
    code = """
    subroutine sum_lo_hi(a, lo, hi, b, res)
      implicit none
      integer, intent(in) :: lo, hi
      real(8), intent(in) :: a(lo:hi)
      real(8), intent(in) :: b(:)
      real(8), intent(out) :: res
      res = a(lo) + a(hi) + sum(b)
    end subroutine sum_lo_hi

    subroutine sum_zero_hi(a, hi, b, res)
      implicit none
      integer, intent(in) :: hi
      real(8), intent(in) :: a(0:hi)
      real(8), intent(in) :: b(:)
      real(8), intent(out) :: res
      res = a(0) + a(hi) + sum(b)
    end subroutine sum_zero_hi
    """

    @pytest.mark.parametrize(("func", "args"), [
        ("sum_lo_hi", ([10.0, 20.0, 30.0], -1, 1, [1.0, 2.0])),
        ("sum_zero_hi", ([10.0, 20.0, 30.0], [1.0, 2.0])),
    ])
    def test_interface_declares_bounds_first(self, func, args):
        assert getattr(self.module, func)(*args) == 43.0
