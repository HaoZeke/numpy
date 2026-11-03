! Declares MPI_INITIALIZED instead of including mpif.h, which MPI
! libraries generate for the Fortran compiler they were configured with
subroutine mpi_initialized_flag(flag)
  logical, intent(out) :: flag
  integer :: ierr
  external mpi_initialized
  call mpi_initialized(flag, ierr)
end subroutine
