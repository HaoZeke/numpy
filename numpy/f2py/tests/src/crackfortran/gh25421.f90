subroutine probe(n, a)
  integer, intent(in) :: n
  real, intent(out) :: a(0:n/2*2)
  a = 1.0
end subroutine
