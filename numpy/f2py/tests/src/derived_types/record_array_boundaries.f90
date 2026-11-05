module record_array_boundaries
  use iso_c_binding, only: c_double, c_int, c_char
  implicit none

  type, bind(c) :: record_atom_c
    real(c_double) :: x
  end type record_atom_c

  type :: record_atom_o
    real(c_double) :: x
  end type record_atom_o

  type, bind(c) :: record_grid_c
    type(record_atom_c) :: plane(2, 3)
    type(record_atom_c) :: cube(2, 2, 3)
  end type record_grid_c

  type :: record_grid_o
    type(record_atom_o) :: plane(2, 3)
    type(record_atom_o) :: cube(2, 2, 3)
  end type record_grid_o

  type :: record_allocated
    type(record_atom_c) :: scalar
    type(record_atom_o), allocatable :: plane(:, :)
    type(record_atom_o), allocatable :: cube(:, :, :)
    type(record_atom_c), allocatable :: interoperable(:, :)
  end type record_allocated

  type :: record_pointer
    type(record_atom_o), pointer :: plane(:, :) => null()
  end type record_pointer

  type :: record_zero
    type(record_atom_c) :: plane(0, 2)
    integer(c_int) :: tag
  end type record_zero

  type(record_atom_o), target, save :: backing(4, 5)

  type, bind(c) :: record_char_atom
    integer(c_int) :: n
    character(kind=c_char) :: mark
  end type record_char_atom

  type, bind(c) :: record_char_grid
    type(record_char_atom) :: plane(2, 3)
  end type record_char_grid

  type :: intrinsic_copy_item
    real(c_double) :: x
  end type intrinsic_copy_item

  type :: intrinsic_copy_box
    type(intrinsic_copy_item) :: scalar
    type(intrinsic_copy_item) :: items(3)
  end type intrinsic_copy_box

  type :: record_len(n)
    integer, len :: n
    type(record_atom_o) :: plane(n, 2)
  end type record_len

  interface assignment(=)
    module procedure shifted_assignment
  end interface

contains

  subroutine shifted_assignment(lhs, rhs)
    type(intrinsic_copy_item), intent(out) :: lhs
    type(intrinsic_copy_item), intent(in) :: rhs
    lhs%x = rhs%x + 50
  end subroutine shifted_assignment

  subroutine fill_intrinsic_copy_box(box)
    type(intrinsic_copy_box), intent(inout) :: box
    integer :: i
    box%scalar%x = 42
    do i = 1, 3
      box%items(i)%x = i
    end do
  end subroutine fill_intrinsic_copy_box

  subroutine fill_grid_c(g)
    type(record_grid_c), intent(inout) :: g
    integer :: i, j, k
    do j = 1, 3
      do i = 1, 2
        g%plane(i, j)%x = 100 * i + j
      end do
    end do
    do k = 1, 3
      do j = 1, 2
        do i = 1, 2
          g%cube(i, j, k)%x = 100 * i + 10 * j + k
        end do
      end do
    end do
  end subroutine fill_grid_c

  subroutine fill_grid_o(g)
    type(record_grid_o), intent(inout) :: g
    integer :: i, j, k
    do j = 1, 3
      do i = 1, 2
        g%plane(i, j)%x = 100 * i + j
      end do
    end do
    do k = 1, 3
      do j = 1, 2
        do i = 1, 2
          g%cube(i, j, k)%x = 100 * i + 10 * j + k
        end do
      end do
    end do
  end subroutine fill_grid_o

  real(c_double) function read_grid_c(g)
    type(record_grid_c), intent(in) :: g
    read_grid_c = g%plane(1, 2)%x + 10 * g%plane(2, 3)%x + g%cube(2, 1, 3)%x
  end function read_grid_c

  real(c_double) function read_grid_o(g)
    type(record_grid_o), intent(in) :: g
    read_grid_o = g%plane(1, 2)%x + 10 * g%plane(2, 3)%x + g%cube(2, 1, 3)%x
  end function read_grid_o

  subroutine attach_records(p)
    type(record_pointer), intent(inout) :: p
    integer :: i, j
    do j = 1, 5
      do i = 1, 4
        backing(i, j)%x = 100 * i + j
      end do
    end do
    p%plane => backing(1:4:2, 2:5:2)
  end subroutine attach_records

  subroutine detach_records(p)
    type(record_pointer), intent(inout) :: p
    nullify(p%plane)
  end subroutine detach_records

  subroutine fill_char_grid(g)
    type(record_char_grid), intent(inout) :: g
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        g%plane(i, j)%n = 100 * i + j
        g%plane(i, j)%mark = 'x'
      end do
    end do
  end subroutine fill_char_grid
end module record_array_boundaries
