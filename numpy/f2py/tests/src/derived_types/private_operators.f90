module default_private_operators
  use iso_c_binding
  implicit none
  private
  public :: private_value, operator(+), operator(-), operator(==), assignment(=)
  public :: public_value_total

  type, bind(c) :: private_value
    integer(c_int) :: value
  end type

  interface operator(+)
    module procedure default_add
  end interface
  interface operator(-)
    module procedure default_negative
  end interface
  interface operator(==)
    module procedure default_equal
  end interface
  interface operator(*)
    module procedure hidden_product
  end interface
  interface assignment(=)
    module procedure default_assign_integer, default_assign_value
  end interface

contains

  function default_add(a, b) result(c)
    type(private_value), intent(in) :: a, b
    type(private_value) :: c
    c%value = a%value + b%value
  end function

  function default_negative(a) result(c)
    type(private_value), intent(in) :: a
    type(private_value) :: c
    c%value = -a%value
  end function

  function default_equal(a, b) result(equal)
    type(private_value), intent(in) :: a, b
    logical :: equal
    equal = a%value == b%value
  end function

  function hidden_product(a, b) result(c)
    type(private_value), intent(in) :: a, b
    type(private_value) :: c
    c%value = a%value * b%value
  end function

  subroutine default_assign_integer(destination, source)
    type(private_value), intent(inout) :: destination
    integer(c_int), intent(in) :: source
    destination%value = destination%value + source
  end subroutine

  subroutine default_assign_value(destination, source)
    type(private_value), intent(out) :: destination
    type(private_value), intent(in) :: source
    destination%value = source%value + 1
  end subroutine

  function public_value_total(value) result(total)
    type(private_value), intent(in) :: value
    integer :: total
    total = value%value
  end function
end module

module explicit_private_operators
  use iso_c_binding
  implicit none
  private :: explicit_add, explicit_assign, operator(-)

  type, bind(c) :: explicit_value
    integer(c_int) :: value
  end type

  interface operator(+)
    module procedure explicit_add
  end interface
  interface operator(-)
    module procedure explicit_negative
  end interface
  interface assignment(=)
    module procedure explicit_assign
  end interface

contains

  function explicit_add(a, b) result(c)
    type(explicit_value), intent(in) :: a, b
    type(explicit_value) :: c
    c%value = a%value + b%value
  end function

  function explicit_negative(a) result(c)
    type(explicit_value), intent(in) :: a
    type(explicit_value) :: c
    c%value = -a%value
  end function

  subroutine explicit_assign(destination, source)
    type(explicit_value), intent(out) :: destination
    integer(c_int), intent(in) :: source
    destination%value = 2 * source
  end subroutine
end module
