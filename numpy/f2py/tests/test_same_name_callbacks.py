import pytest

from . import util


@pytest.mark.slow
class TestSameNameCallbacks(util.F2PyTest):
    suffix = ".f90"
    code = """
module m1
contains
  subroutine apply(f, x, y)
    integer, external :: f
    integer, intent(in) :: x
    integer, intent(out) :: y
    y = f(x) + 1
  end subroutine apply
end module m1

module m2
contains
  subroutine apply(f, x, y)
    real(8), external :: f
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    y = f(x) + 2.0d0
  end subroutine apply
end module m2
"""

    @pytest.mark.parametrize(
        "mod_name,arg,expected",
        [
            ("m1", 5, 11),
            ("m2", 5.0, 12.0),
        ],
    )
    def test_same_name_callbacks(self, mod_name, arg, expected):
        mod = getattr(self.module, mod_name)
        assert mod.apply(lambda t: t * 2, arg) == expected
