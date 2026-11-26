module m
  use iso_c_binding
  implicit none
contains
  subroutine sety(y, x) bind(c, name="sety_c")
    integer(c_int), intent(out) :: y
    integer(c_int), value :: x
    y = x + 1
  end subroutine sety
end module m
