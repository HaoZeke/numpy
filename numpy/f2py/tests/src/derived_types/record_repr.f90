module repr_records
  implicit none
  type :: repr_atom
    real :: x = 0
  end type repr_atom
  type :: repr_container
    type(repr_atom) :: scalar
    type(repr_atom) :: fixed(2, 3)
    type(repr_atom), allocatable :: allocated(:, :)
    type(repr_atom), pointer :: section(:, :) => null()
  end type repr_container
  type(repr_atom), target, save :: backing(3, 4)
contains
  subroutine attach_records(value)
    type(repr_container), intent(inout) :: value
    integer :: i, j
    do j = 1, 4
      do i = 1, 3
        backing(i, j)%x = 10 * i + j
      end do
    end do
    value%section => backing(1:3:2, 2:4:2)
  end subroutine attach_records

  subroutine detach_records(value)
    type(repr_container), intent(inout) :: value
    nullify(value%section)
  end subroutine detach_records
end module repr_records
