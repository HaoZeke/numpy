import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestInterfaceParameter(util.F2PyTest):
    suffix = ".f90"
    code = """
subroutine sub(x, y)
  integer, parameter :: wp = selected_real_kind(15)
  real(wp), intent(in) :: x(:)
  real(wp), intent(out) :: y(size(x))
  y = 2 * x
end subroutine

function fun(x) result(r)
  integer, parameter :: wp = selected_real_kind(15)
  real(wp), intent(in) :: x(:)
  real(wp) :: r(size(x))
  r = 2 * x
end function

subroutine dep_extent(x, y)
  integer, parameter :: p1 = 6
  integer, parameter :: p2 = p1 + 9
  integer, parameter :: wp = selected_real_kind(p2)
  integer, parameter :: n = 3
  integer, parameter :: unused_arr(2) = [2, 3]
  real(wp), intent(in) :: x(:)
  real(wp), intent(out) :: y(n)
  y = unused_arr(1) * x(1:n)
end subroutine

subroutine unused_locals(x, y)
  type :: pair_t
    integer :: a
  end type pair_t
  type(pair_t), parameter :: p = pair_t(2)
  integer, parameter :: arr(2) = [2, 3]
  integer, parameter :: in(2) = [1, 2]
  real(8), intent(in) :: x(:)
  real(8), intent(out) :: y(size(x))
  integer, parameter :: wp = kind(x)
  real(wp), parameter :: eps = epsilon(x)
  real(wp) :: buf(3)
  integer, parameter :: nbuf = size(buf)
  buf = 0.0_wp * eps
  y = (p%a + arr(1) - in(2) + buf(nbuf)) * x
end subroutine

subroutine keyword_names(x, y)
  type :: pair_t
    integer :: a
  end type pair_t
  type(pair_t), parameter :: p = pair_t(2)
  integer, parameter :: wp = selected_real_kind(p=15)
  real(wp), intent(in) :: x(:)
  real(wp), intent(out) :: y(size(x))
  y = p%a * x
end subroutine

subroutine op_eq(x, y)
  type :: box
    integer :: v
  end type box
  type(box), parameter :: eq = box(1)
  real(selected_real_kind(merge(15, 6, 1 .eq. 1))), intent(in) :: x(:)
  real(selected_real_kind(merge(15, 6, 1 .eq. 1))), intent(out) :: y(size(x))
  y = (eq%v + 1) * x
end subroutine

subroutine op_true(x, y)
  type :: box
    integer :: v
  end type box
  type(box), parameter :: true = box(0)
  real(merge(8, 4, .true.)), intent(in) :: x(:)
  real(merge(8, 4, .true.)), intent(out) :: y(size(x))
  y = 2 * x + true%v
end subroutine

subroutine boz_prefix(x, y)
  type :: box
    integer :: v
  end type box
  type(box), parameter :: z = box(0)
  real(selected_real_kind(int(z'f'))), intent(in) :: x(:)
  real(selected_real_kind(int(z'f'))), intent(out) :: y(size(x))
  y = 2 * x + z%v
end subroutine
"""

    @pytest.mark.parametrize(
        "name", ["sub", "fun", "dep_extent", "unused_locals", "keyword_names"]
    )
    def test_interface_parameters(self, name):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(getattr(self.module, name)(x), 2 * x)

    @pytest.mark.parametrize("name", ["op_eq", "op_true", "boz_prefix"])
    def test_operator_and_boz_names(self, name):
        x = np.array([1.0, 2.0, 3.0])
        y = getattr(self.module, name)(x)
        assert y.shape == (3,)
