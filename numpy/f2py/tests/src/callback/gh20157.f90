subroutine gh20157(f, y, r)
  external f
  integer(8) :: f
  integer(8), dimension(:) :: y
  integer(8), intent(out) :: r
  r = f(0)
  r = r + sum(y)
end subroutine gh20157
