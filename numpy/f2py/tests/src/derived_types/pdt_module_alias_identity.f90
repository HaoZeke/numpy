module pdt_alias_owner
  implicit none
  private
  type, public :: alias_cell(n)
    integer, len :: n = 2
    integer :: value = -100
  end type alias_cell
  type(alias_cell(2)), public :: primary
  type(alias_cell(2)), target, public :: pool(2)
  type(alias_cell(2)), pointer, public :: selected(:) => null()
  public :: assignment(=), reset_values, clear_pointer
  interface assignment(=)
    module procedure assign_cell
  end interface assignment(=)
contains
  subroutine assign_cell(lhs, rhs)
    type(alias_cell(*)), intent(out) :: lhs
    type(alias_cell(*)), intent(in) :: rhs
    lhs%value = rhs%value + 30
  end subroutine assign_cell

  subroutine reset_values()
    primary%value = 7
    pool(1)%value = 11
    pool(2)%value = 13
    selected => pool
  end subroutine reset_values

  subroutine clear_pointer()
    nullify(selected)
  end subroutine clear_pointer
end module pdt_alias_owner

module pdt_alias_facade
  use pdt_alias_owner, only: renamed => primary
  implicit none
end module pdt_alias_facade

module pdt_alias_consumer
  use pdt_alias_facade, only: local => renamed
  implicit none
end module pdt_alias_consumer
