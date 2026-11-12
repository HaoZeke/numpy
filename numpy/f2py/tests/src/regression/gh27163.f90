subroutine setn(n)
  use omp_lib
  integer, intent(in) :: n
  call omp_set_num_threads(n)
end subroutine
