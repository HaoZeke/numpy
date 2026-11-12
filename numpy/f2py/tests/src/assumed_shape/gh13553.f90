function trapz(x, y) result(r)
  real*8, dimension(:), intent(in) :: x
  real*8, dimension(size(x)), intent(in) :: y
  real*8 :: r
  integer :: n
  n = size(x)
  r = sum((y(2:n) + y(1:n-1))*(x(2:n) - x(1:n-1)))/2
end function trapz

subroutine scale_by(a, b, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(:)
  real(8), intent(in) :: b(n)
  a(1:n) = a(1:n)*b
end subroutine scale_by

subroutine fill(a, b, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(:)
  real(8), intent(in) :: b(n)
  a(1:n) = b
end subroutine fill

function dsum_impl(x)
  real(8), intent(in) :: x(:)
  real(8) :: dsum_impl
  dsum_impl = sum(x)
end function dsum_impl

subroutine fill_hidden(a, b, n)
  integer, intent(in) :: n
  real(8), intent(inout) :: a(:)
  real(8), intent(in) :: b(n)
  !f2py integer, intent(hide), depend(b) :: n = shape(b, 0)
  a(1:n) = b
end subroutine fill_hidden
