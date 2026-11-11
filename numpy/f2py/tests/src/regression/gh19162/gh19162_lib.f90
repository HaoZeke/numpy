module gh19162_mod
  implicit none
  real(kind=8), allocatable, dimension(:) :: x
contains
  subroutine total(s)
    real(kind=8), intent(out) :: s
    s = 0
    if (allocated(x)) s = sum(x)
  end subroutine total
end module gh19162_mod
