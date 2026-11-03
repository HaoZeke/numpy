module coarray_component
  implicit none

  type :: field
    integer :: id = 0
    real, allocatable :: values(:)[:]
  end type field

  type :: plain
    integer :: n = 0
  end type plain

contains

  subroutine fill(f, n)
    type(field), intent(inout) :: f
    integer, intent(in) :: n
    allocate(f%values(n)[*])
    f%values = 1
  end subroutine fill

  subroutine bump(p)
    type(plain), intent(inout) :: p
    p%n = p%n + this_image()
  end subroutine bump

end module coarray_component
