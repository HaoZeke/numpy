subroutine chk(r8, r4, i1, i2, i4, i8, c)
  implicit none
  real(8), intent(in) :: r8
  real(4), intent(in) :: r4
  integer(1), intent(in) :: i1
  integer(2), intent(in) :: i2
  integer(4), intent(in) :: i4
  integer(8), intent(in) :: i8
  character, intent(in) :: c
  !f2py check(r8 > 0) r8
  !f2py check(r4 > 0) r4
  !f2py check(i1 > 0) i1
  !f2py check(i2 > 0) i2
  !f2py check(i4 > 0) i4
  !f2py check(i8 > 0) i8
  !f2py check(c == 'a') c
end subroutine chk
