subroutine add_renamed(a, b, c)
  use iso_fortran_env, only: dp => real64
  implicit none
  real(kind=dp), intent(in) :: a(2)
  real(kind=dp), intent(in) :: b(2)
  real(kind=dp), intent(out) :: c(2)
  c = a + b
end subroutine add_renamed

subroutine add_intrinsic(a, b, c)
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  real(kind=dp), intent(in) :: a(2)
  real(kind=dp), intent(in) :: b(2)
  real(kind=dp), intent(out) :: c(2)
  c = a + b
end subroutine add_intrinsic
