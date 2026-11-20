! Check that parameter arrays are correctly intercepted.
subroutine foo_array(x, y, z)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter :: pa = 2
  integer, parameter :: intparamarray(2) = (/ 3, 5 /)
  integer, dimension(pa), parameter :: pb = (/ 2, 10 /)
  integer, parameter, dimension(intparamarray(1)) :: pc = (/ 2, 10, 20 /)
  real(dp), parameter :: doubleparamarray(3) = (/ 3.14_dp, 4._dp, 6.44_dp /)
  real(dp), intent(inout) :: x(intparamarray(1))
  real(dp), intent(inout) :: y(intparamarray(2))
  real(dp), intent(out) :: z

  x = x/pb(2)
  y = y*pc(2)
  z = doubleparamarray(1)*doubleparamarray(2) + doubleparamarray(3)

  return
end subroutine

subroutine foo_array_any_index(x, y)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter, dimension(-1:1) :: myparamarray = (/ 6, 3, 1 /)
  integer, parameter, dimension(2) :: nested = (/ 2, 0 /)
  integer, parameter :: dim = 2
  real(dp), intent(in) :: x(myparamarray(-1))
  real(dp), intent(out) :: y(nested(1), myparamarray(nested(dim)))

  y = reshape(x, (/nested(1), myparamarray(nested(2))/))

  return
end subroutine

subroutine foo_array_delims(x)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter, dimension(2) :: myparamarray = (/ (6), 1 /)
  integer, parameter, dimension(3) :: test = (/2, 1, (3)/)
  real(dp), intent(out) :: x

  x = myparamarray(1)+test(3)

  return
end subroutine

subroutine foo_array_expression(x, y, z, w, q, u)
  implicit none
  integer, parameter :: k = 1
  integer, parameter :: dims(2) = (/ 2, 3 /)
  integer, parameter :: idx(2) = (/ 1, 2 /)
  integer, parameter :: lo(2) = (/ -1, 2 /)
  integer, parameter :: a(-1:1) = (/ 5, 6, 7 /)
  integer, parameter :: b(2) = (/ -1, 1 /)
  integer(4), parameter :: kd(2) = (/ 2_4, 3_4 /)
  real, intent(inout) :: x(dims(1) + dims(2))
  real, intent(inout) :: y(dims(idx(k + 1)) * 2)
  real, intent(inout) :: z(2 * (lo(2) - lo(1)))
  real, intent(inout) :: w(a(1 + b(1)))
  real, intent(inout) :: q(dims((1 + 1)) + dims(1))
  real, intent(inout) :: u(kd(1) + kd(2))

  x = x + 1
  y = y + 2
  z = z + 3
  w = w + 4
  q = q + 5
  u = u + 6
end subroutine

module parameter_array_expression_module
  integer, parameter :: mdims(2) = (/ 2, 3 /)
end module

subroutine foo_array_expression_use(x)
  use parameter_array_expression_module, only: mdims
  implicit none
  real, intent(inout) :: x(mdims(1) + mdims(2))

  x = x + 7
end subroutine
