subroutine total(x, n, s)
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  double precision, intent(out) :: s
  s = sum(x)
end subroutine total
