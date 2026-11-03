module lower_bound
  implicit none
contains
  subroutine bounds(x, lb, ub)
    real(8), intent(in) :: x(0:)
    integer, intent(out) :: lb, ub
    lb = lbound(x, 1)
    ub = ubound(x, 1)
  end subroutine bounds

  subroutine bounds_2d(x, lb, ub)
    real(8), intent(in) :: x(0:, -1:)
    integer, intent(out) :: lb(2), ub(2)
    lb = lbound(x)
    ub = ubound(x)
  end subroutine bounds_2d

  subroutine add_index(x, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: x(n:)
    integer :: i
    do i = lbound(x, 1), ubound(x, 1)
      x(i) = x(i) + i
    end do
  end subroutine add_index

  function first_sum(a, c) result(r)
    complex(8), intent(in) :: a (0:), c (0:)
    complex(8) :: r
    r = a(0) + c(0)
  end function first_sum
end module lower_bound
