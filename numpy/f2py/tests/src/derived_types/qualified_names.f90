module namespace_alpha
  use iso_c_binding, only: c_int
  implicit none
  type, bind(c) :: item
    integer(c_int) :: value
  end type item
  interface operator(+)
    module procedure namespace_item_sum
  end interface
  interface assignment(=)
    module procedure namespace_item_assignment
  end interface
contains
  function namespace_item_sum(a, b) result(c)
    type(item), intent(in) :: a, b
    type(item) :: c
    c%value = a%value + b%value
  end function namespace_item_sum
  subroutine namespace_item_assignment(destination, value)
    type(item), intent(out) :: destination
    integer(c_int), intent(in) :: value
    destination%value = value + 3
  end subroutine namespace_item_assignment
  subroutine make_item(value, result)
    integer(c_int), intent(in) :: value
    type(item), intent(out) :: result
    result%value = value
  end subroutine make_item
  function read_item(obj) result(value)
    type(item), intent(in) :: obj
    integer(c_int) :: value
    value = obj%value
  end function read_item
end module namespace_alpha

module namespace_beta
  use iso_c_binding, only: c_double
  implicit none
  type :: item
    real(c_double) :: value
    real(c_double) :: f2py_get_item_value
  end type item
contains
  subroutine make_item(value, result)
    real(c_double), intent(in) :: value
    type(item), intent(out) :: result
    result%value = value
  end subroutine make_item
  function read_item(obj) result(value)
    type(item), intent(in) :: obj
    real(c_double) :: value
    value = obj%value
  end function read_item
  function read_shadow(obj) result(value)
    type(item), intent(in) :: obj
    real(c_double) :: value
    value = obj%f2py_get_item_value
  end function read_shadow
end module namespace_beta

module namespace_join
  use iso_c_binding, only: c_double
  use namespace_alpha, only: alpha_item => item
  use namespace_beta, only: beta_item => item
  implicit none
contains
  function total(a, b) result(value)
    type(alpha_item), intent(in) :: a
    type(beta_item), intent(in) :: b
    real(c_double) :: value
    value = a%value + b%value
  end function total
end module namespace_join

module namespace_poly
  implicit none
  type :: base
    real :: value
  end type base
  type, extends(base) :: child
    real :: extra
  end type child
  type :: unrelated
    integer :: other
  end type unrelated
contains
  function read_base(obj) result(value)
    class(base), intent(in) :: obj
    real :: value
    value = obj%value
  end function read_base
end module namespace_poly
