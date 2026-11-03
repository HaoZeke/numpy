subroutine isuppz_len(compute_v, range, n, il, iu, isuppz, m)
  integer, intent(in) :: compute_v, n, il, iu, m
  character, intent(in) :: range
  integer, intent(out) :: isuppz(m)
  integer :: k
  do k = 1, m
    isuppz(k) = k
  end do
end subroutine isuppz_len
