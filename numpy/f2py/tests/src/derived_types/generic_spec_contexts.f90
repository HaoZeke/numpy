module generic_spec_values
  implicit none
  private
  public :: scope_item, operator ( + ), operator ( - ), assignment ( = )
  public :: operator, assignment
  integer :: operator = 8
  integer :: assignment = 9
  type :: scope_item
    integer :: value
  end type scope_item
  interface operator ( + )
    module procedure :: positive_item
    module procedure sum_items
  end interface
  interface operator ( - )
    module procedure negative_item
  end interface
  interface assignment ( = )
    module procedure assign_integer
  end interface
contains
  function positive_item(argument) result(answer)
    type(scope_item), intent(in) :: argument
    type(scope_item) :: answer
    answer%value = argument%value
  end function positive_item
  function sum_items(left, right) result(answer)
    type(scope_item), intent(in) :: left, right
    type(scope_item) :: answer
    answer%value = left%value + right%value
  end function sum_items
  function negative_item(argument) result(answer)
    type(scope_item), intent(in) :: argument
    type(scope_item) :: answer
    answer%value = -argument%value
  end function negative_item
  subroutine assign_integer(left, right)
    type(scope_item), intent(inout) :: left
    integer, intent(in) :: right
    left%value = right + 50
  end subroutine assign_integer
end module generic_spec_values

module ordinary_generic_words
  implicit none
contains
  integer function operator(value) result(answer)
    integer, intent(in) :: value
    answer = value + 3
  end function operator
  integer function assignment(value) result(answer)
    integer, intent(in) :: value
    answer = value + 5
  end function assignment
end module ordinary_generic_words
