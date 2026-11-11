! numpy/f2py/tests/src/regression/gh24394.f90
subroutine double_it(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = 2.0d0 * x
end subroutine double_it

subroutine double_int(i, j)
  integer, intent(in) :: i
  integer, intent(out) :: j
  j = 2 * i
end subroutine double_int

subroutine double_i8(i, j)
  integer(8), intent(in) :: i
  integer(8), intent(out) :: j
  j = 2 * i
end subroutine double_i8
