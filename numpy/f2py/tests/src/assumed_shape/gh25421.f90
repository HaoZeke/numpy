module gh25421
  implicit none
contains
  subroutine bounds(a, lo, hi)
    complex(kind=8), intent(in) :: a(0:)
    integer, intent(out) :: lo, hi
    lo = lbound(a, 1)
    hi = ubound(a, 1)
  end subroutine bounds

  subroutine fill(b)
    real(kind=8), intent(inout) :: b(-1:, 2:)
    integer :: i, j
    do j = lbound(b, 2), ubound(b, 2)
      do i = lbound(b, 1), ubound(b, 1)
        b(i, j) = 10 * i + j
      end do
    end do
  end subroutine fill

  function total(c) result(s)
    real(kind=8), intent(in) :: c(1:)
    real(kind=8) :: s
    s = sum(c)
  end function total
end module gh25421

subroutine ext_bounds(a, lo, hi)
  implicit none
  real(kind=8), intent(in) :: a(0:)
  integer, intent(out) :: lo, hi
  lo = lbound(a, 1)
  hi = ubound(a, 1)
end subroutine ext_bounds
