module m
  implicit none
contains
  integer function add(x)
    integer, intent(in) :: x
    add = x + 1
  end function add
end module m
