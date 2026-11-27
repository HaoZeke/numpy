subroutine addone(n, a, b)
  implicit none
  integer, intent(in) :: n
  integer(kind=8), intent(inout) :: a(n)
  integer(kind=8), intent(inout) :: b
  integer :: i
  do i = 1, n
    a(i) = a(i) + 1_8
  end do
  b = b + 1_8
end subroutine addone
