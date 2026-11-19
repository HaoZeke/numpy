import pytest

from . import util


@pytest.mark.slow
class TestNarrowBound(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine fill_i1(n, m, a)
  integer(kind=1), intent(in) :: n, m
  real(kind=4), intent(out) :: a(n, m)
  a = 1.0_4
end subroutine fill_i1

subroutine fill_i2(n, m, a)
  integer(kind=2), intent(in) :: n, m
  integer(kind=1), intent(out) :: a(n, m)
  a = 1_1
end subroutine fill_i2
"""

    @pytest.mark.parametrize(("name", "n"), [
        ("fill_i1", 256),
        ("fill_i2", 65541),
    ])
    def test_out_of_range_scalar_rejected(self, name, n):
        with pytest.raises(OverflowError):
            getattr(self.module, name)(n, 2)
