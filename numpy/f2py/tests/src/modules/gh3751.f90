module gh3751_mod
  implicit none
  integer :: counter = 0
  real, allocatable :: values(:)
  real, allocatable, private :: hidden(:)
contains
  integer function values_size()
    if (allocated(values)) then
      values_size = size(values)
    else
      values_size = -1
    end if
  end function values_size

  integer function hidden_size()
    if (allocated(hidden)) then
      hidden_size = size(hidden)
    else
      hidden_size = -1
    end if
  end function hidden_size
end module gh3751_mod

subroutine gh3751_bump()
  integer :: n
  common /gh3751_cb/ n
  n = n + 1
end subroutine gh3751_bump
