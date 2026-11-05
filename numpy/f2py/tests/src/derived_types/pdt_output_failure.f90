module pdt_output_failure
  implicit none
  private :: finish_scalar, finish_array
  integer :: finalized_outputs = 0
  type :: cleanup_cell(k)
    integer, kind :: k = 4
    integer(k) :: tag = 0
  contains
    final :: finish_scalar, finish_array
  end type cleanup_cell
  type(cleanup_cell(4)), allocatable :: selected(:)
contains
  subroutine finish_scalar(value)
    type(cleanup_cell(4)), intent(inout) :: value
    if (value%tag == 777) finalized_outputs = finalized_outputs + 1
  end subroutine finish_scalar

  subroutine finish_array(value)
    type(cleanup_cell(4)), intent(inout) :: value(:)
    finalized_outputs = finalized_outputs + count(value%tag == 777)
  end subroutine finish_array

  subroutine reset_sources()
    if (allocated(selected)) deallocate(selected)
    allocate(selected(2))
    selected%tag = 0
    finalized_outputs = 0
  end subroutine reset_sources

  subroutine scalar_then_invalidate(output, input)
    type(cleanup_cell(4)), allocatable, intent(out) :: output
    type(cleanup_cell(4)), intent(inout) :: input(:)
    allocate(output)
    output%tag = 777
    input%tag = 1
    if (allocated(selected)) deallocate(selected)
  end subroutine scalar_then_invalidate

  subroutine vector_then_invalidate(output, input)
    type(cleanup_cell(4)), allocatable, intent(out) :: output(:)
    type(cleanup_cell(4)), intent(inout) :: input(:)
    allocate(output(2))
    output%tag = 777
    input%tag = 1
    if (allocated(selected)) deallocate(selected)
  end subroutine vector_then_invalidate

  subroutine fixed_then_invalidate(output, input)
    type(cleanup_cell(4)), intent(out) :: output(2)
    type(cleanup_cell(4)), intent(inout) :: input(:)
    output%tag = 777
    input%tag = 1
    if (allocated(selected)) deallocate(selected)
  end subroutine fixed_then_invalidate

  subroutine scalar_success(output)
    type(cleanup_cell(4)), allocatable, intent(out) :: output
    allocate(output)
    output%tag = 777
  end subroutine scalar_success

  subroutine vector_success(output)
    type(cleanup_cell(4)), allocatable, intent(out) :: output(:)
    allocate(output(2))
    output%tag = 777
  end subroutine vector_success
end module pdt_output_failure
