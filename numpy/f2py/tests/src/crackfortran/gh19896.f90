module gh19896
  implicit none

  interface operator(.dot.)
     module procedure dot_real
     module procedure :: dot_int
  end interface operator(.dot.)

  interface operator(+)
     module procedure union
  end interface

  interface assignment(=)
     module procedure int_from_char, real_from_char
  end interface assignment(=)

contains

  function dot_real(a, b) result(c)
    real(8), intent(in) :: a(:), b(:)
    real(8) :: c
    c = sum(a*b)
  end function dot_real

  function dot_int(a, b) result(c)
    integer, intent(in) :: a(:), b(:)
    integer :: c
    c = sum(a*b)
  end function dot_int

  function union(a, b) result(c)
    logical, intent(in) :: a(:), b(:)
    logical :: c(size(a))
    c = a .or. b
  end function union

  subroutine int_from_char(i, s)
    integer, intent(out) :: i
    character(len=*), intent(in) :: s
    read(s, *) i
  end subroutine int_from_char

  subroutine real_from_char(r, s)
    real(8), intent(out) :: r
    character(len=*), intent(in) :: s
    read(s, *) r
  end subroutine real_from_char

end module gh19896
