subroutine alloc_out(n, datbuf)
  implicit none
  integer, intent(in) :: n
  real(8), intent(out), allocatable, dimension(:) :: datbuf
  allocate(datbuf(n))
  datbuf = 1.0d0
end subroutine alloc_out
