subroutine mpi_wrapper_info(major, minor, expected_major, expected_minor, &
                            initialized, initialized_status, version_status, success)
  use mpi
  implicit none
  integer, intent(out) :: major, minor, expected_major, expected_minor
  integer, intent(out) :: initialized_status, version_status, success
  logical, intent(out) :: initialized
  expected_major = MPI_VERSION
  expected_minor = MPI_SUBVERSION
  success = MPI_SUCCESS
  call MPI_INITIALIZED(initialized, initialized_status)
  call MPI_GET_VERSION(major, minor, version_status)
end subroutine mpi_wrapper_info

! Communicator queries require an initialized MPI environment.
subroutine mpi_wrapper_communicator(rank, world_size, rank_status, size_status)
  use mpi
  implicit none
  integer, intent(out) :: rank, world_size, rank_status, size_status
  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, rank_status)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, world_size, size_status)
end subroutine mpi_wrapper_communicator
