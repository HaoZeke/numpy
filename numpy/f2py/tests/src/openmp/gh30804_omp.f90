subroutine count_omp_threads(nwant, nthreads)
   ! Force team size with omp_set_num_threads (gh-30804).
   use omp_lib
   integer, intent(in) :: nwant
   integer, intent(out) :: nthreads
   integer :: local_n
   call omp_set_num_threads(nwant)
   local_n = 1
   !$omp parallel default(none) shared(local_n)
   !$omp single
   local_n = omp_get_num_threads()
   !$omp end single
   !$omp end parallel
   nthreads = local_n
end subroutine count_omp_threads
