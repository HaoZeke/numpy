module fortfuncs
  implicit none
contains
  subroutine square(x,y)
    integer, intent(in) :: x
    integer, intent(out) :: y
    value :: x
    y = x*x
  end subroutine square
end module fortfuncs
