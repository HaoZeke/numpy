module type_only
  implicit none
  type :: pair
    integer :: a, b
  end type pair
  type(pair) :: current
end module type_only
