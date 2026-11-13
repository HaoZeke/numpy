function f(x) bind(c, name="f_c")
  use iso_c_binding
  real(c_double), intent(in) :: x
  real(c_double) :: f
  f = x + 1
end function

subroutine s(x) bind(c, name="s_c")
  use iso_c_binding
  real(c_double), intent(inout) :: x
  x = x + 3
end subroutine
