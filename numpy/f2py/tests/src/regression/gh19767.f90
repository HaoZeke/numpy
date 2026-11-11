! gh-19767: a prefix type must apply to the result variable.
real(kind=8) function mysqrt(x) result(r)
  real(kind=8), intent(in) :: x
  r = x * x
end function mysqrt

character(len=5) function greet() result(r)
  r = 'hello'
end function greet
