module type_index
  implicit none

  ! index is a name that crackfortran renames for variables.
  ! It does not need class, unlike the abstract shape next to it.

  type :: index
    integer :: at = 0
  end type index

contains

  subroutine advance(i)
    type(index), intent(inout) :: i
    i%at = i%at + 1
  end subroutine advance

end module type_index
