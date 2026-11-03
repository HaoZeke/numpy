! A module procedure sees the parameters of its module.
module host
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 3
contains
  subroutine scale_dp(a)
    real(dp), intent(inout) :: a(n)
    a = 2 * a
  end subroutine scale_dp
  subroutine scale_arg(a, n)
    integer, intent(in) :: n
    real(dp), intent(inout) :: a(n)
    a = 2 * a
  end subroutine scale_arg
end module host
