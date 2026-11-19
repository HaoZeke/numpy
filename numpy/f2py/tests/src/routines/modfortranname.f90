module modn
contains
  subroutine incx(x, y)
    integer, intent(in) :: x
    integer, intent(out) :: y
    y = x + 1
  end subroutine incx

  subroutine AddB(a, b, c)
    integer, intent(in) :: a, b
    integer, intent(out) :: c
    c = a + b
  end subroutine AddB
end module modn
