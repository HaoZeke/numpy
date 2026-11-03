module gh19162_use
  use gh19162_mod, only: x, total
  implicit none
!f2py real(kind=8), allocatable, dimension(:) :: x
contains
  subroutine scaled_total(a, s)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(out) :: s
    call total(s)
    s = a*s
  end subroutine scaled_total
end module gh19162_use
