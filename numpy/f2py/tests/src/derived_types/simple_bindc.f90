module simple_types
  use iso_c_binding
  implicit none

  type, bind(c) :: Cartesian
    real(c_double) :: x
    real(c_double) :: y
    real(c_double) :: z
  end type Cartesian

  type :: Point
    real :: x
    real :: y
  end type Point

contains

  subroutine add_scalar(a, b, c) bind(c)
    real(c_double), intent(in) :: a, b
    real(c_double), intent(out) :: c
    c = a + b
  end subroutine add_scalar

end module simple_types
