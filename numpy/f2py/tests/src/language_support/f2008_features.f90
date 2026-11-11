module f2008_features
  implicit none

contains

  subroutine total(a, s)
    real(8), contiguous, intent(in) :: a(:)
    real(8), intent(out) :: s
    integer :: i
    s = 0
    do i = 1, size(a)
      s = s + a(i)
    end do
  end subroutine total

  subroutine fill(n, a)
    integer, intent(in) :: n
    real(8), intent(out) :: a(n)
    integer :: i
    do concurrent (i = 1:n)
      a(i) = i
    end do
  end subroutine fill

end module f2008_features
