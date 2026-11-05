module pdt_module_assignment
  implicit none
  type :: assigned_cell(n)
    integer, len :: n = 2
    integer :: value = 0
  contains
    procedure, private :: assign_cell
    generic, public :: assignment(=) => assign_cell
  end type assigned_cell
  type(assigned_cell(2)) :: fixed(2)
  type(assigned_cell(:)), allocatable :: allocated_cells(:)
contains
  subroutine assign_cell(lhs, rhs)
    class(assigned_cell(*)), intent(inout) :: lhs
    type(assigned_cell(*)), intent(in) :: rhs
    lhs%value = rhs%value + 50
  end subroutine assign_cell

  subroutine reset_values()
    fixed(1)%value = 1
    fixed(2)%value = 2
    if (allocated(allocated_cells)) deallocate(allocated_cells)
  end subroutine reset_values
end module pdt_module_assignment

module pdt_module_generic_assignment
  implicit none
  private
  type, public :: generic_cell(n)
    integer, len :: n = 2
    integer :: value = 0
  end type generic_cell
  type(generic_cell(2)), public :: fixed(2)
  public :: assignment(=), reset_values
  interface assignment(=)
    module procedure assign_cell
  end interface assignment(=)
contains
  subroutine assign_cell(lhs, rhs)
    type(generic_cell(*)), intent(inout) :: lhs
    type(generic_cell(*)), intent(in) :: rhs
    lhs%value = rhs%value + 30
  end subroutine assign_cell

  subroutine reset_values()
    fixed(1)%value = 1
    fixed(2)%value = 2
  end subroutine reset_values
end module pdt_module_generic_assignment
