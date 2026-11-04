module private_operator_owner_left
  use iso_c_binding, only: c_int
  implicit none
  private
  public :: item, operator(+), assignment(=)
  type, bind(c) :: item
    integer(c_int) :: value
  end type item
  interface operator(+)
    module procedure hidden_add
  end interface
  interface operator(*)
    module procedure hidden_product
  end interface
  interface assignment(=)
    module procedure hidden_assign
  end interface
contains
  function hidden_add(a, b) result(c)
    type(item), intent(in) :: a, b
    type(item) :: c
    c%value = a%value + b%value + 0
  end function hidden_add
  function hidden_product(a, b) result(c)
    type(item), intent(in) :: a, b
    type(item) :: c
    c%value = a%value * b%value
  end function hidden_product
  subroutine hidden_assign(destination, source)
    type(item), intent(out) :: destination
    integer(c_int), intent(in) :: source
    destination%value = 2 * source
  end subroutine hidden_assign
end module private_operator_owner_left

module private_operator_owner_right
  use iso_c_binding, only: c_int
  implicit none
  private
  public :: item, operator(+), assignment(=)
  type, bind(c) :: item
    integer(c_int) :: value
  end type item
  interface operator(+)
    module procedure hidden_add
  end interface
  interface operator(*)
    module procedure hidden_product
  end interface
  interface assignment(=)
    module procedure hidden_assign
  end interface
contains
  function hidden_add(a, b) result(c)
    type(item), intent(in) :: a, b
    type(item) :: c
    c%value = a%value + b%value + 100
  end function hidden_add
  function hidden_product(a, b) result(c)
    type(item), intent(in) :: a, b
    type(item) :: c
    c%value = a%value * b%value
  end function hidden_product
  subroutine hidden_assign(destination, source)
    type(item), intent(out) :: destination
    integer(c_int), intent(in) :: source
    destination%value = 3 * source
  end subroutine hidden_assign
end module private_operator_owner_right
