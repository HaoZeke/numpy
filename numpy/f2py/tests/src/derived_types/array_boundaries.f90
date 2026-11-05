module array_boundary_types
  use iso_c_binding, only: c_double, c_int
  implicit none

  type, bind(c) :: boundary_bindc
    real(c_double) :: plane(2, 3)
    real(c_double) :: cube(2, 3, 4)
  end type boundary_bindc

  type :: boundary_opaque
    real(c_double) :: plane(2, 3)
    real(c_double) :: cube(2, 3, 4)
    real(c_double), allocatable :: dynamic_plane(:, :)
  end type boundary_opaque

  type :: boundary_len(n)
    integer, len :: n
    real(c_double) :: plane(n, 3)
    real(c_double) :: offset_plane(n + 1, 3)
  end type boundary_len

  type :: boundary_atom
    real(c_double) :: x
  end type boundary_atom

  type :: boundary_pointer
    real(c_double), pointer :: plane(:, :) => null()
  end type boundary_pointer

  type :: boundary_zero
    real(c_double) :: plane(0, 3)
    integer(c_int) :: tag
  end type boundary_zero

  real(c_double), target, save :: pointer_storage(4, 5)

contains

  subroutine attach_numeric_section(p)
    type(boundary_pointer), intent(inout) :: p
    integer :: i, j
    do j = 1, 5
      do i = 1, 4
        pointer_storage(i, j) = 100 * i + j
      end do
    end do
    p%plane => pointer_storage(1:4:2, 2:5:2)
  end subroutine attach_numeric_section

  subroutine fill_bindc(p)
    type(boundary_bindc), intent(inout) :: p
    integer :: i, j, k
    do j = 1, 3
      do i = 1, 2
        p%plane(i, j) = 100 * i + j
        do k = 1, 4
          p%cube(i, j, k) = 100 * i + 10 * j + k
        end do
      end do
    end do
  end subroutine fill_bindc

  subroutine fill_opaque(p)
    type(boundary_opaque), intent(inout) :: p
    integer :: i, j, k
    do j = 1, 3
      do i = 1, 2
        p%plane(i, j) = 100 * i + j
        do k = 1, 4
          p%cube(i, j, k) = 100 * i + 10 * j + k
        end do
      end do
    end do
  end subroutine fill_opaque

  real(c_double) function read_bindc(p)
    type(boundary_bindc), intent(in) :: p
    read_bindc = p%plane(1, 2) + 10 * p%plane(2, 3) + p%cube(2, 1, 4)
  end function read_bindc

  real(c_double) function read_opaque(p)
    type(boundary_opaque), intent(in) :: p
    read_opaque = p%plane(1, 2) + 10 * p%plane(2, 3) + p%cube(2, 1, 4)
  end function read_opaque

  real(c_double) function inspect_atoms(a, n, hook)
    integer, intent(in) :: n
    type(boundary_atom), intent(in) :: a(n)
    real(c_double), external :: hook
    integer :: i
    inspect_atoms = hook(0.0_c_double)
    do i = 1, n
      inspect_atoms = inspect_atoms + a(i)%x
    end do
  end function inspect_atoms
end module array_boundary_types
