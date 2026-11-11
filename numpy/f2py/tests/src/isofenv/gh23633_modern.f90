subroutine fib(a, n)
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n
  real(real64), intent(out) :: a(n)
  integer :: i

  if (n >= 1) a(1) = 0.0_real64
  if (n >= 2) a(2) = 1.0_real64
  do i = 3, n
    a(i) = a(i - 1) + a(i - 2)
  end do
end subroutine fib
