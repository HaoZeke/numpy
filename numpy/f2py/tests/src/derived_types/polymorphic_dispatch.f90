! Test polymorphic CLASS(T) dispatch (F2018 7.3.2.3)
module poly_dispatch_mod
  implicit none

  type :: shape
    real :: area = 0.0
  end type shape

  type, extends(shape) :: circle
    real :: radius = 0.0
  end type circle

  type, extends(shape) :: rectangle
    real :: width = 0.0
    real :: height = 0.0
  end type rectangle

contains

  subroutine compute_area(s)
    class(shape), intent(inout) :: s
    ! Base implementation: area already set or zero
    select type (s)
    type is (circle)
      s%area = 3.14159265 * s%radius * s%radius
    type is (rectangle)
      s%area = s%width * s%height
    class default
      ! Keep existing area
    end select
  end subroutine compute_area

  function get_area(s) result(a)
    class(shape), intent(in) :: s
    real :: a
    a = s%area
  end function get_area

end module poly_dispatch_mod
