subroutine incr(n, a, b, c, d, s)
  implicit none
  integer, intent(in) :: n
  integer(1), intent(inout) :: a(n)
  integer(2), intent(inout) :: b(n)
  integer(4), intent(inout) :: c(n)
  integer(8), intent(inout) :: d(n)
  integer(8), intent(inout) :: s
  a = a + 1_1
  b = b + 1_2
  c = c + 1_4
  d = d + 1_8
  s = s + 1_8
end subroutine incr
