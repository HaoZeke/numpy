module u
  use p
  integer, parameter :: scale = 2
  character(len=n) :: name = "hello"
  character(len=m) :: tiny = "ab"
  character(len=2*m) :: doubled = "abcd"
  character(len=scale*m) :: scaled = "wxyz"
  character*(n) :: starred = "world"
contains
  subroutine get_name(s)
    character(len=n), intent(out) :: s
    s = "hello"
  end subroutine

  subroutine echo(n, s)
    integer, intent(in) :: n
    character(len=n), intent(out) :: s
    s = "ab"
  end subroutine
end module
