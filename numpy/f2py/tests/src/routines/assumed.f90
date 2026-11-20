subroutine shape_sum(a, s)
  real(8), intent(in) :: a(:)
  real(8), intent(out) :: s
  s = sum(a)
end subroutine

subroutine SumArr(a, s)
  real(8), intent(in) :: a(:)
  real(8), intent(out) :: s
  s = sum(a)
end subroutine
