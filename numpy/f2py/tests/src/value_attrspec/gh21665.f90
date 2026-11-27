module fortfuncs
  implicit none
contains
  subroutine square(x,y)
    integer, intent(in), value :: x
    integer, intent(out) :: y
    y = x*x
  end subroutine square

  subroutine square_complex(x, y)
    complex, intent(in), value :: x
    complex, intent(out) :: y
    y = x*x
  end subroutine square_complex

  integer function square_fn(x)
    integer, intent(in), value :: x
    square_fn = x*x
  end function square_fn
end module fortfuncs
