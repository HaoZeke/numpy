function twice(x) result(y)
  use iso_fortran_env, only: real64
  implicit none
  real(real64), intent(in) :: x
  real(real64) :: y
  y = 2 * x
end function twice
