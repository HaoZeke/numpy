module record_source_c
  use iso_c_binding, only: c_double
  implicit none
  type, bind(c) :: item
    real(c_double) :: x
  end type item
end module record_source_c

module record_source_o
  use iso_c_binding, only: c_double
  implicit none
  type :: item
    real(c_double) :: x
  end type item
end module record_source_o

module imported_record_components
  use iso_c_binding, only: c_double
  use record_source_c, only: c_record => item
  use record_source_o, only: o_record => item
  implicit none
  private :: c_record, o_record

  type, bind(c) :: imported_c_grid
    type(c_record) :: plane(2, 3)
    type(c_record) :: cube(2, 2, 3)
  end type imported_c_grid

  type :: imported_o_grid
    type(o_record) :: plane(2, 3)
    type(o_record) :: cube(2, 2, 3)
  end type imported_o_grid

  type :: imported_alloc_grid
    type(o_record), allocatable :: plane(:, :)
    type(c_record), allocatable :: cube(:, :, :)
  end type imported_alloc_grid

contains

  subroutine fill_imported_c(g)
    type(imported_c_grid), intent(inout) :: g
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
  end subroutine fill_imported_c

  subroutine fill_imported_o(g)
    type(imported_o_grid), intent(inout) :: g
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
  end subroutine fill_imported_o

  real(c_double) function read_imported_c(g)
    type(imported_c_grid), intent(in) :: g
    read_imported_c = g%plane(1, 2)%x + 10 * g%plane(2, 3)%x + g%cube(2, 1, 3)%x
  end function read_imported_c

  real(c_double) function read_imported_o(g)
    type(imported_o_grid), intent(in) :: g
    read_imported_o = g%plane(1, 2)%x + 10 * g%plane(2, 3)%x + g%cube(2, 1, 3)%x
  end function read_imported_o
end module imported_record_components

module imported_record_only
  use record_source_c, only: item
  implicit none
  type, bind(c) :: only_record_grid
    type(item) :: cube(2, 3, 4)
  end type only_record_grid
end module imported_record_only

module imported_record_reexport
  use record_source_o, only: renamed_item => item
  implicit none
end module imported_record_reexport

module imported_record_transitive
  use imported_record_reexport, only: local_item => renamed_item
  implicit none
  type :: transitive_record_grid
    type(local_item) :: plane(2, 3)
  end type transitive_record_grid
end module imported_record_transitive

module imported_nested_owner
  use record_source_o, only: component_atom => item
  implicit none
  private :: component_atom
  type :: nested_record_item
    type(component_atom) :: records(2, 3)
  end type nested_record_item
end module imported_nested_owner

module imported_nested_consumer
  use iso_c_binding, only: c_double
  use imported_nested_owner, only: nested_alias => nested_record_item
  implicit none
  private :: nested_alias
  type :: nested_record_grid
    type(nested_alias) :: plane(2, 2)
  end type nested_record_grid
contains
  real(c_double) function read_nested(g)
    type(nested_record_grid), intent(in) :: g
    read_nested = g%plane(2, 1)%records(1, 3)%x
  end function read_nested
end module imported_nested_consumer
