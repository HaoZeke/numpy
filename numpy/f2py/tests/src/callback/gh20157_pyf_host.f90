function pyf_host(f, y) result(r)
  external f
  integer(8) :: r, f
  integer(8), dimension(:) :: y
  r = f(0) + sum(y)
end function pyf_host
