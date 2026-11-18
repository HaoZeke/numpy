import pytest

from . import util


@pytest.mark.slow
class TestModIntDim(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine mod_extent(n, m, a)
        integer, intent(in) :: n, m
        integer, intent(out) :: a(mod(n, m) + 5)
        a = mod(n, m)
      end subroutine mod_extent

      subroutine int_extent(x, a)
        real, intent(in) :: x
        integer, intent(out) :: a(int(x))
        a = int(x)
      end subroutine int_extent
    """

    @pytest.mark.parametrize(
        ("routine", "args", "extent", "value"),
        [
            ("mod_extent", (-7, 4), 2, -3),
            ("int_extent", (3.7,), 3, 3),
        ],
    )
    def test_extent(self, routine, args, extent, value):
        a = getattr(self.module, routine)(*args)
        assert a.shape == (extent,)
        assert int(a[0]) == value
