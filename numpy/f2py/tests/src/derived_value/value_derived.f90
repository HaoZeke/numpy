module mvalue
  use iso_c_binding
  implicit none

  type :: pair
    integer :: a
  end type pair

  type, bind(c) :: cpair
    integer(c_int) :: a
  end type cpair
contains
  subroutine module_value(x)
    type(pair), value :: x
  end subroutine module_value

  subroutine module_bindc_value(x) bind(c)
    type(cpair), value :: x
  end subroutine module_bindc_value

  subroutine add_one(k, y)
    integer, intent(in) :: k
    integer, intent(out) :: y
    y = k + 1
  end subroutine add_one
end module mvalue

subroutine free_value(x)
  implicit none

  type :: pair
    integer :: a
  end type pair

  type(pair), value :: x
end subroutine free_value
