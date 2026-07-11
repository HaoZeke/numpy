! gh-20103: contained procedures must not leak into the interface
subroutine outer20103(x, y)
  implicit none
  real(8), intent(in) :: x
  real(8), intent(out) :: y
  y = helper(x) + 1.0d0
contains
  function helper(a) result(r)
    real(8), intent(in) :: a
    real(8) :: r
    r = a * 2.0d0
  end function helper
end subroutine outer20103
