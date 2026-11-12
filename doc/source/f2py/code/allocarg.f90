subroutine positive(x, n, idx)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: x(n)
  integer, allocatable, intent(out) :: idx(:)
  integer :: i, k
  allocate(idx(count(x > 0)))
  k = 0
  do i = 1, n
    if (x(i) > 0) then
      k = k + 1
      idx(k) = i
    end if
  end do
end subroutine positive
