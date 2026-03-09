module opaque_types
  implicit none
  private
  public :: Point, distance

  type :: Point
    real :: x
    real :: y
  end type Point

contains

  function distance(p1, p2) result(d)
    type(Point), intent(in) :: p1, p2
    real :: d
    d = sqrt((p1%x - p2%x)**2 + (p1%y - p2%y)**2)
  end function distance

end module opaque_types
