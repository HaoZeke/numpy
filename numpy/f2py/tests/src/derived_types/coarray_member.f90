! Test coarray component wrapping (F2018 7.5.4.3)
! Local access works without coarray runtime.
module coarray_mod
  implicit none

  type :: distributed_data
    integer :: id = 0
    real, allocatable :: values(:)
    ! Note: coarray member would be declared as:
    ! real, allocatable, codimension[*] :: shared_val
    ! But we test with a regular type first since coarray
    ! syntax requires compiler support.
  end type distributed_data

contains

  subroutine init_data(d, id, n)
    type(distributed_data), intent(inout) :: d
    integer, intent(in) :: id, n
    integer :: i
    d%id = id
    if (allocated(d%values)) deallocate(d%values)
    allocate(d%values(n))
    do i = 1, n
      d%values(i) = real(id * 100 + i)
    end do
  end subroutine init_data

  function sum_values(d) result(s)
    type(distributed_data), intent(in) :: d
    real :: s
    if (allocated(d%values)) then
      s = sum(d%values)
    else
      s = 0.0
    end if
  end function sum_values

end module coarray_mod
