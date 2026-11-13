subroutine echo2(a, b, n, m)
  implicit none
  integer, intent(in) :: n, m
  real(8), intent(in) :: a(n, m)
  real(8), intent(out) :: b(n, m)
  b = a
end subroutine echo2

subroutine echo3(a, b, n, m, p)
  implicit none
  integer, intent(in) :: n, m, p
  real(8), intent(in) :: a(n, m, p)
  real(8), intent(out) :: b(n, m, p)
  b = a
end subroutine echo3

subroutine bump2(a, n, m)
  implicit none
  integer, intent(in) :: n, m
  real(8), intent(inout) :: a(n, m)
  a = a + 1.0d0
end subroutine bump2

subroutine bump3(a, n, m, p)
  implicit none
  integer, intent(in) :: n, m, p
  real(8), intent(inout) :: a(n, m, p)
  a = a + 1.0d0
end subroutine bump3
