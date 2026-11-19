import pytest

from . import util


@pytest.mark.slow
class TestIsoCParamKind(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine add_real(x, y)
  use iso_c_binding, only: c_double
  implicit none
  integer, parameter :: dp = c_double
  real(dp), intent(in) :: x
  real(dp), intent(out) :: y
  y = x + 1
end subroutine add_real

subroutine add_int(x, y)
  use iso_c_binding, only: c_int64_t
  implicit none
  integer, parameter :: ik = c_int64_t
  integer(ik), intent(in) :: x
  integer(ik), intent(out) :: y
  y = x + 1
end subroutine add_int
"""

    @pytest.mark.parametrize(
        "name, value, expected",
        [
            ("add_real", 1.0 + 2**-24, 2.0 + 2**-24),
            ("add_int", 1 << 40, (1 << 40) + 1),
        ],
    )
    def test_param_kind(self, name, value, expected):
        got = getattr(self.module, name)(value)
        assert got == expected
