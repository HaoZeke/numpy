! Two assumed-shape hosts both using local callback name f (gh-20157 isolation).
function host_i8(f, y) result(r)
  external f
  integer(8) :: r, f
  integer(8), dimension(:) :: y
  r = f(0) + sum(y)
end function host_i8

function host_dp(f, y) result(r)
  external f
  double precision :: r, f
  double precision, dimension(:) :: y
  r = f(0d0) + sum(y)
end function host_dp
