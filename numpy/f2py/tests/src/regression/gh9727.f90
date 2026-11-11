! gh-9727: real(kind=16) intent(out) must not be passed to Py_BuildValue.
subroutine get_pi(pi)
  implicit none
  real(kind=16), intent(out) :: pi
  pi = 4.0_16 * atan(1.0_16)
end subroutine get_pi

function get_e() result(e)
  implicit none
  real(kind=16) :: e
  e = 1.0_16
end function get_e
