! gh-19767: module-less function must return its computed value
function mysqrt(x) result(r)
  implicit none
  real(kind=8), intent(in) :: x
  real(kind=8) :: r
  r = sqrt(x)
end function mysqrt
