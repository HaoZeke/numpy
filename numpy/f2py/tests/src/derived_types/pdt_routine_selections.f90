module pdt_routine_selections
  use iso_c_binding, only: c_double
  implicit none
  type :: routine_cell(n)
    integer, len :: n = 2
    real(c_double) :: value
    real(c_double) :: values(n)
  end type routine_cell
  type(routine_cell(2)) :: single
  type(routine_cell(2)) :: fixed(2, 3)
contains
  subroutine reset_values()
    single%value = 11
    single%values = [1, 2]
    fixed%value = 17
  end subroutine reset_values

  subroutine bump(value, amount)
    type(routine_cell(*)), intent(inout) :: value
    real(c_double), intent(in) :: amount
    value%value = value%value + amount
  end subroutine bump

  subroutine sum_pair(left, right, total)
    type(routine_cell(*)), intent(in) :: left, right
    real(c_double), intent(out) :: total
    total = left%value + right%value
  end subroutine sum_pair

  subroutine bump_grid(values, amount)
    type(routine_cell(*)), intent(inout) :: values(:, :)
    real(c_double), intent(in) :: amount
    values%value = values%value + amount
  end subroutine bump_grid

  subroutine make_grid(n, m, length, values)
    integer, intent(in) :: n, m, length
    type(routine_cell(:)), allocatable, intent(out) :: values(:, :)
    allocate(routine_cell(length) :: values(-1:n-2, 3:m+2))
    values%value = 23
  end subroutine make_grid

  subroutine make_cell(length, value)
    integer, intent(in) :: length
    type(routine_cell(length)), intent(out) :: value
    value%value = 29
  end subroutine make_cell

  function cell_result(length) result(value)
    integer, intent(in) :: length
    type(routine_cell(length)) :: value
    value%value = 31
  end function cell_result

  subroutine optional_read(value, total)
    type(routine_cell(*)), optional, intent(in) :: value
    real(c_double), intent(out) :: total
    total = -1
    if (present(value)) total = value%value
  end subroutine optional_read

  subroutine fixed_length(value)
    type(routine_cell(2)), intent(inout) :: value
    value%value = 47
  end subroutine fixed_length

  subroutine resize_cell(value, length)
    type(routine_cell(:)), allocatable, intent(inout) :: value
    integer, intent(in) :: length
    if (allocated(value)) deallocate(value)
    allocate(routine_cell(length) :: value)
    value%value = 61
  end subroutine resize_cell

  subroutine make_fixed_grid(values)
    type(routine_cell(n=2)), allocatable, intent(out) :: values(:,:)
    integer :: i, j
    allocate(values(2,3))
    do j = 1, 3
      do i = 1, 2
        values(i,j)%value = 101
        values(i,j)%values = 103
      end do
    end do
  end subroutine make_fixed_grid

  subroutine make_fixed_cell(value)
    type(routine_cell(n=2)), allocatable, intent(out) :: value
    allocate(value)
    value%value = 107
    value%values = 109
  end subroutine make_fixed_cell

end module pdt_routine_selections
