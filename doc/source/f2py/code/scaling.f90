module scaling
  use store, only: x
  implicit none
  !f2py real(kind=8), allocatable, dimension(:) :: x
contains
  subroutine scale(a)
    real(kind=8), intent(in) :: a
    if (allocated(x)) x = a*x
  end subroutine scale
end module scaling
