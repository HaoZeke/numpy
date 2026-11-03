subroutine call_two(fun, r)
  implicit none
  integer, external :: fun
  integer, intent(out) :: r
  integer :: a, b
  a = 1
  b = 2
  r = fun(a, b)
end subroutine call_two
