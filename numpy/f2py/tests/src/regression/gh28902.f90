subroutine rank_of(comm, rank)
  implicit none
  include 'mpif.h'
  integer, intent(in) :: comm
  integer, intent(out) :: rank
  integer ierr
  call MPI_Comm_rank(comm, rank, ierr)
end subroutine
