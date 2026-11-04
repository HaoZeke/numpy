module sequence_mod
  implicit none

  type :: seq_point
    sequence
    double precision :: x, y, z
  end type seq_point

  type :: seq_with_array
    sequence
    integer :: n
    double precision :: data(3)
  end type seq_with_array

contains

  subroutine translate(p, dx, dy, dz)
    type(seq_point), intent(inout) :: p
    double precision, intent(in) :: dx, dy, dz
    p%x = p%x + dx
    p%y = p%y + dy
    p%z = p%z + dz
  end subroutine translate

  function distance(a, b) result(d)
    type(seq_point), intent(in) :: a, b
    double precision :: d
    d = sqrt((a%x - b%x)**2 + (a%y - b%y)**2 + (a%z - b%z)**2)
  end function distance

end module sequence_mod
