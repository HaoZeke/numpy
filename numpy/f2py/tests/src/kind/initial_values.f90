! Initial values of optional arguments, which f2py writes in C
subroutine scale(x, tol, n, y)
  real(8), intent(in) :: x, tol
  integer, intent(in) :: n
  real(8), intent(out) :: y
  !f2py real(8), optional :: tol = 1.0d-6
  !f2py integer, optional :: n = 3_8
  y = x * tol * n
end subroutine scale
