! Test iso_fortran_env parameter resolution in derived types
module iso_env_type_mod
  use iso_fortran_env, only: int32, int64, real64
  implicit none

  type :: precise_point
    integer(int32) :: id
    real(real64) :: x, y, z
  end type precise_point

  type :: big_counter
    integer(int64) :: count = 0
  end type big_counter

contains

  subroutine set_point(p, id, x, y, z)
    type(precise_point), intent(inout) :: p
    integer(int32), intent(in) :: id
    real(real64), intent(in) :: x, y, z
    p%id = id
    p%x = x
    p%y = y
    p%z = z
  end subroutine set_point

  function point_distance(a, b) result(d)
    type(precise_point), intent(in) :: a, b
    real(real64) :: d
    d = sqrt((a%x - b%x)**2 + (a%y - b%y)**2 + (a%z - b%z)**2)
  end function point_distance

end module iso_env_type_mod
