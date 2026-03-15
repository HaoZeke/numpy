module abstract_type_mod
  implicit none

  ! Abstract type for testing (F2018 7.5.7.1)
  ! Uses 'BaseShape' to avoid conflict with numpy's 'shape' builtin
  type, abstract :: BaseShape
    double precision :: area
  contains
    procedure(compute_area_iface), deferred :: compute_area
  end type BaseShape

  abstract interface
    subroutine compute_area_iface(self)
      import :: BaseShape
      class(BaseShape), intent(inout) :: self
    end subroutine
  end interface

  type, extends(BaseShape) :: Disk
    double precision :: radius
  contains
    procedure :: compute_area => disk_compute_area
  end type Disk

contains
  subroutine disk_compute_area(self)
    class(Disk), intent(inout) :: self
    self%area = 3.14159265358979d0 * self%radius * self%radius
  end subroutine
end module abstract_type_mod
