! Host-local parameter kind must resolve in nested host interface (gh-20157).
function host_local_dp(f, y) result(r)
  integer, parameter :: dp = kind(1.0d0)
  external f
  real(dp) :: r, f
  real(dp), dimension(:) :: y
  r = f(0.0_dp) + sum(y)
end function host_local_dp
