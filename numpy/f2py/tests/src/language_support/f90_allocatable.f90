module f90_allocatable
  implicit none

  real(8), allocatable :: buffer(:)

contains

  integer function buffer_size()
    if (allocated(buffer)) then
      buffer_size = size(buffer)
    else
      buffer_size = -1
    end if
  end function buffer_size

end module f90_allocatable
