subroutine addi(n, x)
  use iso_c_binding, only: c_int
  integer(c_int), value :: n
  integer(c_int), intent(inout) :: x(:)
  x = x + n
end subroutine addi

subroutine addd(n, x)
  use iso_c_binding, only: c_double
  real(c_double), value :: n
  real(c_double), intent(inout) :: x(:)
  x = x + n
end subroutine addd
