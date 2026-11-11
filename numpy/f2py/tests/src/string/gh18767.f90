subroutine string_len(s, n, last)
  implicit none
  character(len=*), intent(in) :: s
  integer(kind=8), intent(out) :: n
  integer, intent(out) :: last
  n = len(s, kind=8)
  last = ichar(s(n:n))
end subroutine string_len

subroutine string_array_len(a, m, n, last)
  implicit none
  integer, intent(in) :: m
  character(len=*), intent(in) :: a(m)
  integer(kind=8), intent(out) :: n
  integer, intent(out) :: last
  n = len(a, kind=8)
  last = ichar(a(m)(n:n))
end subroutine string_array_len
