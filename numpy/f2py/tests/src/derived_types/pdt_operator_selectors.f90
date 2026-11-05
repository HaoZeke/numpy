module pdt_operator_values
  implicit none
  integer, parameter :: wide_kind = kind(0.0d0)
  type :: op_cell(k)
    integer, kind :: k = kind(0.0)
    real(k) :: value
  end type op_cell
  type(op_cell(4)) :: single
  type(op_cell(4)) :: fixed(-1:0, 3:4)
  type(op_cell(4)), target :: target_cell
  type(op_cell(4)), pointer :: pointer_cell => null()
  type(op_cell(4)), protected :: locked
  type(op_cell(wide_kind)) :: wide
  private :: add_cells, positive_cell, negative_cell, compare_cells, assign_cell
  interface operator(+)
    module procedure add_cells, positive_cell, add_integer, integer_add
    module procedure mixed_cells
  end interface
  interface operator(-)
    module procedure negative_cell
  end interface
  interface operator(==)
    module procedure compare_cells
  end interface
  interface operator(*)
    module procedure product_value
  end interface
  interface operator(/)
    module procedure pointer_quotient
  end interface
  interface assignment(=)
    module procedure assign_cell
  end interface
contains
  subroutine reset_values()
    single%value = 7
    fixed%value = 9
    target_cell%value = 11
    pointer_cell => target_cell
    locked%value = 23
    wide%value = 19
  end subroutine reset_values
  function add_cells(left, right) result(value)
    type(op_cell(4)), intent(in) :: left, right
    type(op_cell(4)) :: value
    value%value = left%value + right%value
  end function add_cells
  function positive_cell(left) result(value)
    type(op_cell(4)), intent(in) :: left
    type(op_cell(4)) :: value
    value%value = left%value
  end function positive_cell
  function negative_cell(left) result(value)
    type(op_cell(4)), intent(in) :: left
    type(op_cell(4)) :: value
    value%value = -left%value
  end function negative_cell
  function add_integer(left, right) result(value)
    type(op_cell(4)), intent(in) :: left
    integer, intent(in) :: right
    type(op_cell(4)) :: value
    value%value = left%value + right
  end function add_integer
  function integer_add(left, right) result(value)
    integer, intent(in) :: left
    type(op_cell(4)), intent(in) :: right
    type(op_cell(4)) :: value
    value%value = left + right%value
  end function integer_add
  function mixed_cells(left, right) result(value)
    type(op_cell(4)), intent(in) :: left
    type(op_cell(wide_kind+0)), intent(in) :: right
    type(op_cell(wide_kind+0)) :: value
    value%value = left%value + right%value
  end function mixed_cells
  logical function compare_cells(left, right)
    type(op_cell(4)), intent(in) :: left, right
    compare_cells = left%value == right%value
  end function compare_cells
  integer function product_value(left, right)
    type(op_cell(4)), intent(in) :: left, right
    product_value = int(left%value * right%value)
  end function product_value
  integer function pointer_quotient(left, right)
    type(op_cell(4)), pointer, intent(in) :: left, right
    pointer_quotient = int(left%value / right%value)
  end function pointer_quotient
  subroutine assign_cell(left, right)
    type(op_cell(4)), intent(inout) :: left
    type(op_cell(4)), intent(in) :: right
    left%value = -999
    left%value = right%value + 50
  end subroutine assign_cell
end module pdt_operator_values

module pdt_operator_facade
  use pdt_operator_values, only: renamed_single => single
  implicit none
end module pdt_operator_facade

module pdt_operator_lengths
  implicit none
  type :: len_cell(n)
    integer, len :: n = 2
    integer :: value
    integer :: values(n)
  end type len_cell
  type(len_cell(2)) :: single
  type(len_cell(2)), target :: target_cell
  type(len_cell(:)), pointer :: deferred_pointer => null()
  type(len_cell(2)), protected :: locked
  interface operator(-)
    module procedure negate_cell
  end interface
  interface operator(+)
    module procedure add_fixed
  end interface
  interface operator(==)
    module procedure allocated_fixed_equal
  end interface
  interface operator(*)
    module procedure assumed_pointer_total
  end interface
  interface operator(/)
    module procedure deferred_pointer_total
  end interface
contains
  subroutine reset_values()
    single%value = 7
    single%values = [8, 9]
    target_cell%value = 7
    target_cell%values = [8, 9]
    deferred_pointer => target_cell
    locked%value = 23
    locked%values = [24, 25]
  end subroutine reset_values
  function negate_cell(left) result(value)
    type(len_cell(*)), intent(in) :: left
    type(len_cell(:)), allocatable :: value
    allocate(len_cell(n=left%n) :: value)
    value%value = -left%value
    value%values = -left%values
  end function negate_cell
  function add_fixed(left, right) result(value)
    type(len_cell(2)), intent(in) :: left
    integer, intent(in) :: right
    type(len_cell(2)) :: value
    value%value = left%value + right
    value%values = left%values + right
  end function add_fixed
  logical function allocated_fixed_equal(left, right)
    type(len_cell(2)), allocatable, intent(in) :: left
    integer, intent(in) :: right
    allocated_fixed_equal = left%value + sum(left%values) == right
  end function allocated_fixed_equal
  integer function assumed_pointer_total(left, right)
    type(len_cell(*)), pointer, intent(in) :: left
    integer, intent(in) :: right
    assumed_pointer_total = left%value + sum(left%values) + right
  end function assumed_pointer_total
  integer function deferred_pointer_total(left, right)
    type(len_cell(:)), pointer, intent(in) :: left
    integer, intent(in) :: right
    deferred_pointer_total = left%value + sum(left%values) + right
  end function deferred_pointer_total
end module pdt_operator_lengths

module pdt_operator_allocations
  implicit none
  type :: allocation_cell(n)
    integer, len :: n = 2
    integer :: value
    integer :: values(n)
  end type allocation_cell
  type(allocation_cell(:)), allocatable :: destination
  type(allocation_cell(2)) :: fixed
  interface assignment(=)
    module procedure assign_allocation
  end interface
  interface operator(*)
    module procedure allocation_sum
  end interface
contains
  subroutine reset_values()
    if (allocated(destination)) deallocate(destination)
    allocate(allocation_cell(n=2) :: destination)
    destination%value = 7
    destination%values = [8, 9]
    fixed%value = 17
    fixed%values = [18, 19]
  end subroutine reset_values
  subroutine assign_allocation(left, right)
    type(allocation_cell(:)), allocatable, intent(inout) :: left
    type(allocation_cell(*)), intent(in) :: right
    if (allocated(left)) deallocate(left)
    if (right%value < 0) return
    allocate(allocation_cell(n=right%n+1) :: left)
    left%value = right%value + 50
    left%values(:right%n) = right%values
    left%values(right%n+1) = 99
  end subroutine assign_allocation
  integer function allocation_sum(left, right)
    type(allocation_cell(:)), allocatable, intent(in) :: left
    integer, intent(in) :: right
    allocation_sum = left%value + sum(left%values) + right
  end function allocation_sum
end module pdt_operator_allocations

module pdt_operator_other
  implicit none
  type :: op_cell(k)
    integer, kind :: k = kind(0.0)
    real(k) :: value
  end type op_cell
  type(op_cell(4)) :: single
  interface operator(-)
    module procedure negate_other
  end interface
contains
  subroutine reset_values()
    single%value = 31
  end subroutine reset_values
  function negate_other(left) result(value)
    type(op_cell(4)), intent(in) :: left
    type(op_cell(4)) :: value
    value%value = -left%value
  end function negate_other
end module pdt_operator_other
