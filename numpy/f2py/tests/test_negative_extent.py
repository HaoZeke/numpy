import textwrap

from . import util


class TestNegativeExtent(util.F2PyTest):
    suffix = ".f90"
    code = textwrap.dedent("""
        subroutine out_neg(a, s)
          integer, intent(out) :: a(-1)
          integer, intent(out) :: s
          s = size(a)
        end subroutine out_neg

        subroutine out_rev(a, s)
          integer, intent(out) :: a(3:1)
          integer, intent(out) :: s
          s = size(a)
        end subroutine out_rev

        subroutine out_grid(a, s)
          integer, intent(out) :: a(5:3, 4)
          integer, intent(out) :: s
          s = size(a)
        end subroutine out_grid
        """)

    def test_negative_constant_extent(self):
        a, s = self.module.out_neg()
        assert a.shape == (0,)
        assert s == 0
        a, s = self.module.out_rev()
        assert a.shape == (0,)
        assert s == 0
        a, s = self.module.out_grid()
        assert a.shape == (0, 4)
        assert s == 0
