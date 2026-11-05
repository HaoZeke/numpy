subroutine init_zero_common()
  implicit none
  character(len=0) :: scalar_value
  character(len=16) :: scalar_guard
  character(len=0) :: array_values(3)
  character(len=16) :: array_guard
  common /zero_scalar/ scalar_value, scalar_guard
  common /zero_array/ array_values, array_guard
  scalar_guard = 'abcdefghijklmnop'
  array_guard = 'abcdefghijklmnop'
end subroutine init_zero_common

module zero_char_mod
  implicit none
  character(len=0), allocatable :: values(:)
contains
  subroutine storage_shape(count, width)
    integer, intent(out) :: count, width
    count = size(values)
    width = len(values)
  end subroutine storage_shape
end module zero_char_mod
