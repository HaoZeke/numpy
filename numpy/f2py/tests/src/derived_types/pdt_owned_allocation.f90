module pdt_owned_allocation
  implicit none
  type :: allocation_cell(k)
    integer, kind :: k = 4
    integer(k) :: value
  end type allocation_cell
  type(allocation_cell(4)), allocatable :: allocated_unit
contains
  subroutine reset_values()
    if (allocated(allocated_unit)) deallocate(allocated_unit)
  end subroutine reset_values

  subroutine make_unit()
    if (allocated(allocated_unit)) deallocate(allocated_unit)
    allocate(allocated_unit)
    allocated_unit%value = 67
  end subroutine make_unit

  subroutine clear_unit(value)
    type(allocation_cell(4)), allocatable, intent(inout) :: value
    if (allocated(value)) deallocate(value)
  end subroutine clear_unit

  subroutine replace_unit(value, amount)
    type(allocation_cell(4)), allocatable, intent(inout) :: value
    integer, intent(in) :: amount
    if (allocated(value)) deallocate(value)
    allocate(value)
    value%value = amount
  end subroutine replace_unit
end module pdt_owned_allocation
