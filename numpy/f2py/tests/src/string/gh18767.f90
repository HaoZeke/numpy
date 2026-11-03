subroutine string_len(s, n, last)
  implicit none
  character(len=*), intent(in) :: s
  integer(kind=8), intent(out) :: n
  integer, intent(out) :: last
  n = len(s, kind=8)
  last = ichar(s(n:n))
end subroutine string_len
