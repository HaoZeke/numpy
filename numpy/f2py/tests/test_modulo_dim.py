import pytest

from . import util


@pytest.mark.slow
class TestModuloDim(util.F2PyTest):
    suffix = ".f90"
    code = """
      subroutine modulo_extent(n, m, a)
        integer, intent(in) :: n, m
        integer, intent(out) :: a(modulo(n, m) + 5)
        a = modulo(n, m)
      end subroutine modulo_extent
    """

    @pytest.mark.parametrize(
        ("n", "m", "rem"),
        [
            (-7, 4, 1),
            (7, -4, -1),
        ],
    )
    def test_extent(self, n, m, rem):
        a = self.module.modulo_extent(n, m)
        assert a.shape == (rem + 5,)
        assert int(a[0]) == rem
