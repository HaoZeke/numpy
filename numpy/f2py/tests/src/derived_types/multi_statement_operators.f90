module multi_statement_operators
  use iso_c_binding
  implicit none
  private
  public :: list_value, operator(+), add

  type, bind(c) :: list_value
    integer(c_int) :: value
  end type

  interface operator(+)
    module procedure :: positive_value
    module procedure add_values
  end interface
  interface add
    module procedure :: add_values
  end interface

contains

  function positive_value(argument) result(answer)
    type(list_value), intent(in) :: argument
    type(list_value) :: answer
    answer%value = argument%value
  end function

  function add_values(left, right) result(answer)
    type(list_value), intent(in) :: left, right
    type(list_value) :: answer
    answer%value = left%value + right%value
  end function
end module multi_statement_operators
