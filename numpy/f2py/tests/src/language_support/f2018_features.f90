subroutine no_implicit(x, y)
  implicit none (type, external)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = 2 * x
end subroutine no_implicit
