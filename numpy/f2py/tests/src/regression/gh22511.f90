! gh-22511: PARAMETER addresses must not be stack temporaries.
module gh22511
  implicit none
  integer, parameter :: my_const = 1234
  real(kind=8), parameter :: pi = 3.0d0
  integer, parameter :: arr(2) = (/2, 4/)
  character(len=5), parameter :: word = 'hello'
  integer :: my_var = 7
contains
  subroutine cross(a, b, c)
    real, intent(in) :: a(3)
    real, intent(in) :: b(3)
    real, intent(out) :: c(3)
    c(1) = a(2) * b(3) - a(3) * b(2)
    c(2) = a(3) * b(1) - a(1) * b(3)
    c(3) = a(1) * b(2) - a(2) * b(1)
  end subroutine cross
end module gh22511
