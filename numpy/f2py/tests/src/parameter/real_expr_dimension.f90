module real_expr_consts
  implicit none
  real, parameter :: big = 1.0e2
  integer, parameter :: n = int(big)
  integer, parameter :: m = int(2.5 * 4)
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: dp_precision = selected_real_kind(precision(1.0d0))
end module real_expr_consts

module real_expr_dimension
  use real_expr_consts, only: n, m
  implicit none
contains
  subroutine fill(x, y)
    real, intent(out) :: x(n), y(m)
    x = 1
    y = 2
  end subroutine fill
end module real_expr_dimension
