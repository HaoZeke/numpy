module pdt_module_values
  use iso_c_binding, only: c_double
  implicit none
  type :: cell(k, n)
    integer, kind :: k = c_double
    integer, len :: n = 2
    real(k) :: value
    real(k) :: values(n)
  end type cell
  type(cell(c_double, 2)) :: single
  type(cell(c_double, 2)) :: fixed(-1:0, 3:5)
  type(cell(c_double, 2)) :: cube(0:1, 2:3, -2:0)
  type(cell(c_double, :)), allocatable :: optional
  type(cell(c_double, :)), allocatable :: allocated_cells(:, :)
  type(cell(c_double, 2)), allocatable :: fixed_allocated(:, :)
  type(cell(c_double, 2)), target :: pool(-1:0, 3:5)
  type(cell(c_double, 2)), pointer :: selected(:, :) => null()
  type(cell(c_double, 2)), protected :: locked
  type(cell(c_double, 2)), private :: hidden
contains
  subroutine reset_values()
    integer :: i, j, l
    single%value = 7
    single%values = [8, 9]
    locked%value = 23
    locked%values = [24, 25]
    hidden%value = -1
    hidden%values = 0
    do j = 3, 5
      do i = -1, 0
        fixed(i,j)%value = 100 * (i + 2) + (j - 2)
        fixed(i,j)%values = [fixed(i,j)%value, -fixed(i,j)%value]
        pool(i,j)%value = fixed(i,j)%value
        pool(i,j)%values = fixed(i,j)%values
      end do
    end do
    do l = -2, 0
      do j = 2, 3
        do i = 0, 1
          cube(i,j,l)%value = 100 * (i + 1) + 10 * (j - 1) + l + 3
          cube(i,j,l)%values = [cube(i,j,l)%value, -cube(i,j,l)%value]
        end do
      end do
    end do
    if (allocated(optional)) deallocate(optional)
    if (allocated(allocated_cells)) deallocate(allocated_cells)
    if (allocated(fixed_allocated)) deallocate(fixed_allocated)
    nullify(selected)
  end subroutine reset_values

  subroutine allocate_optional(n)
    integer, intent(in) :: n
    if (allocated(optional)) deallocate(optional)
    allocate(cell(c_double, n) :: optional)
    optional%value = n
    optional%values = n
  end subroutine allocate_optional

  subroutine resize_cells(n, m, length)
    integer, intent(in) :: n, m, length
    if (allocated(allocated_cells)) deallocate(allocated_cells)
    allocate(cell(c_double, length) :: allocated_cells(-2:n-3, 4:m+3))
    allocated_cells%value = 17
  end subroutine resize_cells

  subroutine associate_pool()
    selected => pool
  end subroutine associate_pool

  real(c_double) function scalar_value()
    scalar_value = single%value
  end function scalar_value

  subroutine bump_cell(value, amount)
    type(cell(c_double, *)), intent(inout) :: value
    real(c_double), intent(in) :: amount
    value%value = value%value + amount
  end subroutine bump_cell

  real(c_double) function fixed_value(i, j)
    integer, intent(in) :: i, j
    fixed_value = fixed(i,j)%value
  end function fixed_value
end module pdt_module_values

module pdt_module_facade
  use pdt_module_values, only: renamed_single => single
  implicit none
end module pdt_module_facade
