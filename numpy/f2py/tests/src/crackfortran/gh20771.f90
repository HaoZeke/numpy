subroutine isuppz_len(compute_v, range, n, il, iu, isuppz, m)
  integer, intent(in) :: compute_v, n, il, iu, m
  character, intent(in) :: range
  integer, intent(out) :: isuppz(m)
  integer :: k
  do k = 1, m
    isuppz(k) = k
  end do
end subroutine isuppz_len

subroutine vl_len(v, vl, m)
  double precision, intent(in) :: v
  integer, intent(in) :: m
  double precision, intent(out) :: vl(m)
  integer :: k
  do k = 1, m
    vl(k) = k
  end do
end subroutine vl_len
