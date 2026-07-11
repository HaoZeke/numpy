subroutine norm2_r64(x, n, res)
  use iso_fortran_env, only: real64
  implicit none
  integer, intent(in) :: n
  real(real64), intent(in) :: x(n)
  real(real64), intent(out) :: res
  res = sqrt(sum(x**2))
end subroutine norm2_r64
