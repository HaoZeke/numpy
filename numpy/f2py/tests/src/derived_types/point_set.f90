module point_set
  implicit none

  type :: point
    real :: x = 0.0
    real :: y = 0.0
    real :: z = 0.0
  end type point

contains

  subroutine set_point(p, x, y, z)
    type(point), intent(inout) :: p
    real, intent(in) :: x, y, z
    p%x = x
    p%y = y
    p%z = z
  end subroutine set_point

end module point_set
