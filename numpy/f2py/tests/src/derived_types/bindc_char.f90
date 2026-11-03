module bindc_char
  use iso_c_binding, only: c_char, c_int
  implicit none

  type, bind(c) :: tagged
    integer(c_int) :: n = 0
    character(len=1, kind=c_char) :: mark = 'a'
  end type tagged

  ! characters without a length have length 1
  type :: letters
    character :: first = 'x'
    character(kind=c_char) :: second = 'y'
  end type letters

contains

  subroutine bump(t)
    type(tagged), intent(inout) :: t
    t%n = t%n + 1
    t%mark = 'b'
  end subroutine bump

  subroutine swap(l)
    type(letters), intent(inout) :: l
    character :: c
    c = l%first
    l%first = l%second
    l%second = c
  end subroutine swap

end module bindc_char
