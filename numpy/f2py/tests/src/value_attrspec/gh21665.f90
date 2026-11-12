module fortfuncs
  implicit none
contains
  subroutine square(x,y)
    integer, intent(in), value :: x
    integer, intent(out) :: y
    y = x*x
  end subroutine square

  subroutine square_real(x,y)
    real, intent(in), value :: x
    real, intent(out) :: y
    y = x*x
  end subroutine square_real
end module fortfuncs
