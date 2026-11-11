! gh-33220: a backslash does not escape a closing quote.
subroutine probe(n, got)
  implicit none
  character(len=*), intent(in) :: n
  integer, intent(out) :: got
  character(len=len('c:\'//n)) :: m
  got = len(m)
end subroutine probe
