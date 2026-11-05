subroutine echo_runtime_strings(values, result, width)
  implicit none
  character(len=*), intent(in) :: values(:)
  character(len=len(values)), intent(out) :: result(size(values))
  integer, intent(out) :: width
  result = values
  width = len(values)
end subroutine echo_runtime_strings

subroutine zero_allocation_state(is_allocated, count, width)
  use zero_char_mod, only: values
  implicit none
  integer, intent(out) :: is_allocated, count, width
  is_allocated = 0
  count = -1
  width = -1
  if (allocated(values)) then
    is_allocated = 1
    count = size(values)
    width = len(values)
  end if
end subroutine zero_allocation_state
