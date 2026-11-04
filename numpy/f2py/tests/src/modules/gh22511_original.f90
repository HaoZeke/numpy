module f2py_test_module

  integer, public, parameter :: my_const = 1234
  public :: cross

contains

  ! Simple function to calculate vector cross product for 3-D vectors.
  SUBROUTINE cross(a, b, c)

    REAL, INTENT(IN) :: a(3)
    REAL, INTENT(IN) :: b(3)
    REAL, INTENT(OUT) :: c(3)

    c(1) = a(2) * b(3) - a(3) * b(2)
    c(2) = a(3) * b(1) - a(1) * b(3)
    c(3) = a(1) * b(2) - a(2) * b(1)
    RETURN
  END

end module f2py_test_module
