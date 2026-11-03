subroutine omp_team_size(nreq, nteam)
   ! The omp_lib procedures have explicit interfaces and are defined in
   ! the shared OpenMP runtime library
   use omp_lib, only: omp_set_dynamic, omp_set_num_threads, &
                      omp_get_num_threads
   integer, intent(in) :: nreq
   integer, intent(out) :: nteam
   call omp_set_dynamic(.false.)
   call omp_set_num_threads(nreq)
   nteam = 0
   !$omp parallel
   !$omp single
   nteam = omp_get_num_threads()
   !$omp end single
   !$omp end parallel
end subroutine omp_team_size
