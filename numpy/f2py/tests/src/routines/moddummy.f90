module modd
contains
  subroutine realinc(x, y)
    integer, intent(in) :: x
    integer, intent(out) :: y
    y = x + 1
  end subroutine realinc
end module modd
