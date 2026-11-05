module allocation_status
  implicit none
  integer, allocatable :: numbers(:,:)
  character(len=3), allocatable :: words(:)
  character(len=0), allocatable :: zeros(:,:)
contains
  subroutine allocated_state(state)
    integer, intent(out) :: state(3)
    state = 0
    if (allocated(numbers)) state(1) = 1
    if (allocated(words)) state(2) = 1
    if (allocated(zeros)) state(3) = 1
  end subroutine allocated_state
end module allocation_status
