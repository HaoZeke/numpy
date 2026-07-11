! gh-24394: scalar converter deprecation tests
subroutine double_it(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = 2.0d0 * x
end subroutine double_it
