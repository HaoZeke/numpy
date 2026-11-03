module type_names
  implicit none

  ! shape and index are names that crackfortran renames for variables
  type, abstract :: shape
  contains
    procedure(area_iface), deferred :: area
  end type shape

  abstract interface
    function area_iface(self) result(a)
      import :: shape
      class(shape), intent(in) :: self
      double precision :: a
    end function area_iface
  end interface

  type, extends(shape) :: square
    double precision :: side = 1
  contains
    procedure :: area => square_area
  end type square

  type :: index
    integer :: at = 0
  end type index

contains

  function square_area(self) result(a)
    class(square), intent(in) :: self
    double precision :: a
    a = self%side**2
  end function square_area

  function total_area(s) result(a)
    class(shape), intent(in) :: s
    double precision :: a
    a = s%area()
  end function total_area

  subroutine advance(i)
    type(index), intent(inout) :: i
    i%at = i%at + 1
  end subroutine advance

end module type_names
