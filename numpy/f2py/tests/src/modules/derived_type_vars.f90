module typed
  implicit none
  type :: pair
    integer :: a, b
  end type pair
  type(pair), parameter :: origin = pair(0, 0)
  type(pair) :: current
  integer :: plain = 5
contains
  function total(n)
    integer, intent(in) :: n
    integer :: total
    total = n + plain
  end function total
end module typed
