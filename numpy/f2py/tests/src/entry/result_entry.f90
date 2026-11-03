function f(x) result(r)
  double precision, intent(in) :: x
  double precision :: r
  integer :: g
  r = 2 * x
  return
entry g(x)
  g = int(3 * x)
  return
entry h(x)
  h = 4 * x
end function f
