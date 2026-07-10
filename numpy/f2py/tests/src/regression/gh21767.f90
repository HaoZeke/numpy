! gh-21767: fortran function wrappers pickle by reference
subroutine double_it(x, y)
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = 2.0d0 * x
end subroutine double_it
