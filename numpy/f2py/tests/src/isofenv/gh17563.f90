subroutine pyfunc(i, x)
  use iso_fortran_env, only: dp => real64, int32
  implicit none
  integer(int32), intent(in) :: i
  real(dp), intent(out) :: x
  x = real(i, dp) / 3
end subroutine pyfunc
