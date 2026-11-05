subroutine probe(work, m, n, got)
  implicit none
  integer :: m, n, got
  real :: work(m)
  got = m
  if (m > 0) work(1) = real(n)
end subroutine probe
