! gh-13553: assumed-shape + size(x) dimension; pyf-first wrapper must not
! emit f2py-only attributes (depend/check/required) into Fortran interfaces.
function trapz(x, y) result(r)
  real*8, dimension(:), intent(in)  :: x
  real*8, dimension(size(x)), intent(in)  :: y
  real*8              :: r
  integer :: n
  n = size(x)
  r = sum((y(2:n) + y(1:n-1))*(x(2:n) - x(1:n-1)))/2
end function trapz
